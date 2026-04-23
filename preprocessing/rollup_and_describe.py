#!/usr/bin/env python3
"""Roll up raw MIMIC event codes and build code descriptions."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import tempfile
import csv
import zipfile
import time
from pathlib import Path

import duckdb
import requests

LOGGER = logging.getLogger("rollup_and_describe")

PHE_ICD9_URL = "https://phewascatalog.org/files/phecode_icd9_rolled.csv"
PHE_ICD10_URL = "https://phewascatalog.org/files/phecode_icd10.csv"
PHE_DEF_ZIP_URL = "https://phewascatalog.org/files/phecode_definitions1.2.csv.zip"
CCS10_URL = "https://www.hcup-us.ahrq.gov/toolssoftware/ccs10/ccs_pr_icd10pcs_2019_1.zip"
CCS9_URL = "https://hcup-us.ahrq.gov/toolssoftware/ccs/Single_Level_CCS_2015.zip"
RXNORM_FULL_URL = "https://download.nlm.nih.gov/rxnorm/RxNorm_full_prescribe_04062026.zip"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roll up event codes and build descriptions.")
    parser.add_argument("--test_mode", action="store_true", help="Use _test files instead of _full.")
    parser.add_argument(
        "--force-rxnorm",
        action="store_true",
        help="Force RxNorm zip download and RRF re-parse even when mapping caches exist.",
    )
    return parser.parse_args()


def _execute_csv_sql(con: duckdb.DuckDBPyConnection, sql_template: str, filepath: Path) -> None:
    escaped = str(filepath).replace("'", "''")
    direct_source = f"read_csv_auto('{escaped}')"
    try:
        con.execute(sql_template.format(source=direct_source))
        return
    except duckdb.IOException as exc:
        if "GZIP" not in str(exc).upper():
            raise

    with tempfile.TemporaryDirectory() as tmp_dir:
        fifo_path = Path(tmp_dir) / "stream.csv"
        os.mkfifo(fifo_path)
        proc = subprocess.Popen(
            ["bash", "-lc", f"gzip -dc \"{filepath}\" > \"{fifo_path}\""],
            stderr=subprocess.PIPE,
        )
        try:
            escaped_fifo = str(fifo_path).replace("'", "''")
            fifo_source = f"read_csv_auto('{escaped_fifo}')"
            con.execute(sql_template.format(source=fifo_source))
        finally:
            _, stderr = proc.communicate()
            if proc.returncode not in (0, 2):
                stderr_text = stderr.decode("utf-8", errors="ignore") if stderr else ""
                raise RuntimeError(f"gzip stream failed for {filepath}: {stderr_text}")


def _validate_downloaded_file(destination: Path, url: str) -> None:
    if not destination.exists():
        raise FileNotFoundError(f"Expected file missing after download: {destination} (url={url})")
    size = destination.stat().st_size
    if size <= 1024:
        raise ValueError(
            f"Downloaded file at {destination} is too small ({size} bytes); "
            f"likely an HTML error page, not real data. url={url}"
        )
    if destination.suffix.lower() == ".zip" and not zipfile.is_zipfile(destination):
        raise ValueError(f"File at {destination} is not a valid zip archive. url={url}")


def ensure_download(url: str, destination: Path) -> Path:
    if destination.exists() and destination.stat().st_size > 0:
        try:
            _validate_downloaded_file(destination, url)
            return destination
        except ValueError as exc:
            LOGGER.warning("Removing invalid cached download %s: %s", destination, exc)
            destination.unlink(missing_ok=True)
    elif destination.exists() and destination.stat().st_size == 0:
        destination.unlink(missing_ok=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading %s", url)
    try:
        with requests.get(
            url,
            timeout=120,
            stream=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; TALE-EHR-preprocessing/1.0)"},
        ) as resp:
            resp.raise_for_status()
            with destination.open("wb") as out:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        out.write(chunk)
        _validate_downloaded_file(destination, url)
    except Exception as exc:
        if destination.exists():
            destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"Download failed or validation failed for {url} (destination={destination})"
        ) from exc
    return destination


def extract_zip(zip_path: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            target = out_dir / Path(name).name
            with zf.open(name) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(target)
    return extracted


def resolve_mimic_root(repo_root: Path) -> Path:
    raw_dir = repo_root / "data" / "raw"
    candidate = raw_dir / "mimiciv-3.1"
    if candidate.exists():
        resolved = candidate.resolve()
        LOGGER.info("Resolved MIMIC-IV root from symlink: %s", resolved)
        return resolved
    raise FileNotFoundError(f"Could not resolve MIMIC-IV root from {raw_dir}")


def load_json(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): str(v) for k, v in data.items()}


def save_json(path: Path, data: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True, indent=2)


def save_json_fast(path: Path, data: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True)


def _dict_to_duckdb_table(
    con: duckdb.DuckDBPyConnection,
    data: dict[str, str],
    table_name: str,
    col_names: tuple[str, str],
) -> None:
    col0, col1 = col_names
    con.execute(f"DROP TABLE IF EXISTS {table_name}")
    if not data:
        con.execute(f"CREATE TABLE {table_name}({col0} VARCHAR, {col1} VARCHAR)")
        return

    tmp_name = f"__tmp_{table_name}"
    registered = False
    try:
        import pyarrow as pa

        arrow_tbl = pa.table(
            {
                col0: pa.array(list(data.keys()), type=pa.string()),
                col1: pa.array(list(data.values()), type=pa.string()),
            }
        )
        con.register(tmp_name, arrow_tbl)
        registered = True
    except Exception:
        import pandas as pd

        df = pd.DataFrame({col0: list(data.keys()), col1: list(data.values())})
        con.register(tmp_name, df)
        registered = True

    try:
        con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM {tmp_name}")
    finally:
        if registered:
            try:
                con.unregister(tmp_name)
            except Exception:
                pass


def normalize_code(code: str) -> str:
    return code.strip().upper().replace(".", "")


def create_phecode_maps(con: duckdb.DuckDBPyConnection, mappings_dir: Path) -> None:
    def _build_from_csv(icd9_csv: Path, icd10_csv: Path, defs_csv: Path | None) -> None:
        _execute_csv_sql(
            con,
            """
            CREATE OR REPLACE TABLE phe_icd9 AS
            SELECT
                REPLACE(UPPER(TRIM(CAST(ICD9 AS VARCHAR))), '.', '') AS icd_code_norm,
                MIN(TRIM(CAST(PheCode AS VARCHAR))) AS phecode
            FROM {source}
            WHERE ICD9 IS NOT NULL AND PheCode IS NOT NULL
            GROUP BY REPLACE(UPPER(TRIM(CAST(ICD9 AS VARCHAR))), '.', '')
            """,
            icd9_csv,
        )
        _execute_csv_sql(
            con,
            """
            CREATE OR REPLACE TABLE phe_icd10 AS
            SELECT
                REPLACE(UPPER(TRIM(CAST(ICD10 AS VARCHAR))), '.', '') AS icd_code_norm,
                MIN(TRIM(CAST(PheCode AS VARCHAR))) AS phecode
            FROM {source}
            WHERE ICD10 IS NOT NULL AND PheCode IS NOT NULL
            GROUP BY REPLACE(UPPER(TRIM(CAST(ICD10 AS VARCHAR))), '.', '')
            """,
            icd10_csv,
        )
        if defs_csv is not None and defs_csv.exists():
            _execute_csv_sql(
                con,
                """
                CREATE OR REPLACE TABLE phe_defs AS
                SELECT
                    TRIM(CAST(phecode AS VARCHAR)) AS phecode,
                    CAST(phenotype AS VARCHAR) AS phenotype,
                    CAST(category AS VARCHAR) AS category
                FROM {source}
                WHERE phecode IS NOT NULL
                """,
                defs_csv,
            )
        else:
            con.execute(
                """
                CREATE OR REPLACE TABLE phe_defs AS
                SELECT DISTINCT
                    phecode,
                    phecode AS phenotype,
                    'PheWAS' AS category
                FROM (
                    SELECT phecode FROM phe_icd9
                    UNION
                    SELECT phecode FROM phe_icd10
                )
                """
            )

    try:
        icd9_csv = ensure_download(PHE_ICD9_URL, mappings_dir / "phecode_icd9_rolled.csv")
        icd10_csv = ensure_download(PHE_ICD10_URL, mappings_dir / "phecode_icd10.csv")
        phe_zip = ensure_download(PHE_DEF_ZIP_URL, mappings_dir / "phecode_definitions1.2.csv.zip")
        extracted = extract_zip(phe_zip, mappings_dir / "phe_defs")
        phe_def_csv = next((p for p in extracted if p.suffix.lower() == ".csv"), None)
        if phe_def_csv is None:
            raise FileNotFoundError("Could not find phecode definitions CSV after extracting zip")
        _build_from_csv(icd9_csv, icd10_csv, phe_def_csv)
        LOGGER.info("Loaded PheCode maps from official PheWAS catalog files.")
    except Exception as exc:
        LOGGER.warning("Primary PheWAS catalog download failed (%s). Trying alternate standard sources.", exc)
        try:
            icd10_csv = ensure_download(
                "https://raw.githubusercontent.com/PheWAS/PheWAS/master/data/phecode_icd10.csv",
                mappings_dir / "phecode_icd10.csv",
            )
            icd9_csv = ensure_download(
                "https://raw.githubusercontent.com/PheWAS/PheWAS/master/data/phecode_icd9_rolled.csv",
                mappings_dir / "phecode_icd9_rolled.csv",
            )
            phe_defs_csv = ensure_download(
                "https://raw.githubusercontent.com/PheWAS/PheWAS/master/data/phecode_definitions1.2.csv",
                mappings_dir / "phecode_definitions1.2.csv",
            )
            _build_from_csv(icd9_csv, icd10_csv, phe_defs_csv)
            LOGGER.info("Loaded PheCode maps from PheWAS GitHub CSV mirror.")
        except Exception as exc2:
            LOGGER.warning("CSV mirror unavailable (%s). Building standard numeric mappings from official PheWAS .rda files.", exc2)
            import pyreadr
            import pandas as pd

            map_rda = ensure_download(
                "https://raw.githubusercontent.com/PheWAS/PheWAS/master/data/phecode_map.rda",
                mappings_dir / "phecode_map.rda",
            )
            map10_rda = ensure_download(
                "https://raw.githubusercontent.com/PheWAS/PheWAS/master/data/phecode_map_icd10.rda",
                mappings_dir / "phecode_map_icd10.rda",
            )
            info_rda = ensure_download(
                "https://raw.githubusercontent.com/PheWAS/PheWAS/master/data/pheinfo.rda",
                mappings_dir / "pheinfo.rda",
            )

            map_df = pyreadr.read_r(str(map_rda))["phecode_map"]
            map10_df = pyreadr.read_r(str(map10_rda))["phecode_map_icd10"]
            info_df = pyreadr.read_r(str(info_rda))["pheinfo"]

            icd9_df = map_df[map_df["vocabulary_id"].astype(str).str.upper().isin(["ICD9", "ICD9CM"])][["code", "phecode"]].copy()
            icd9_df.columns = ["ICD9", "PheCode"]
            icd9_df = icd9_df.dropna().drop_duplicates()
            icd10_df = map10_df[map10_df["vocabulary_id"].astype(str).str.upper().str.contains("ICD10")][["code", "phecode"]].copy()
            icd10_df.columns = ["ICD10", "PheCode"]
            icd10_df = icd10_df.dropna().drop_duplicates()
            defs_df = info_df[["phecode", "description", "group"]].copy()
            defs_df.columns = ["phecode", "phenotype", "category"]
            defs_df = defs_df.dropna().drop_duplicates(subset=["phecode"])

            icd9_csv = mappings_dir / "phecode_icd9_rolled.csv"
            icd10_csv = mappings_dir / "phecode_icd10.csv"
            defs_csv = mappings_dir / "phecode_definitions1.2.csv"
            icd9_df.to_csv(icd9_csv, index=False)
            icd10_df.to_csv(icd10_csv, index=False)
            defs_df.to_csv(defs_csv, index=False)

            _build_from_csv(icd9_csv, icd10_csv, defs_csv)
            LOGGER.info("Loaded PheCode maps from standard numeric PheWAS .rda extraction.")


def find_existing_ndc_rxnorm_files() -> list[Path]:
    roots = [
        Path("/home/suraj/Git/MIMIC-IV-Data-Pipeline"),
        Path("/home/suraj/Git/MIMIC-IV-Data-Pipeline-main"),
    ]
    files: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        files.extend(root.rglob("*ndc*"))
        files.extend(root.rglob("*rxnorm*"))
    return [p for p in files if p.is_file()]


def _read_candidate_mapping(con: duckdb.DuckDBPyConnection, path: Path) -> tuple[dict[str, str], dict[str, str]]:
    try:
        escaped = str(path).replace("'", "''")
        df = con.execute(f"SELECT * FROM read_csv_auto('{escaped}') LIMIT 500000").df()
    except Exception:
        return {}, {}
    if df.empty:
        return {}, {}

    cols_lower = {c.lower(): c for c in df.columns}
    proc_code_col = None
    ccs_col = None
    desc_col = None

    for c in df.columns:
        cl = c.lower()
        if proc_code_col is None and ("icd" in cl or "code" in cl):
            proc_code_col = c
        if ccs_col is None and "ccs" in cl:
            ccs_col = c
        if desc_col is None and ("label" in cl or "description" in cl or "desc" in cl):
            desc_col = c

    if proc_code_col is None or ccs_col is None:
        # Last-chance exact name fallback.
        proc_code_col = cols_lower.get("icd_code") or cols_lower.get("code")
        ccs_col = cols_lower.get("ccs") or cols_lower.get("ccs_category")
    if proc_code_col is None or ccs_col is None:
        return {}, {}

    map_df = df[[proc_code_col, ccs_col] + ([desc_col] if desc_col else [])].copy()
    map_df = map_df.dropna(subset=[proc_code_col, ccs_col])
    ccs_map: dict[str, str] = {}
    ccs_desc: dict[str, str] = {}
    for _, row in map_df.iterrows():
        raw = normalize_code(str(row[proc_code_col]))
        ccs_code = str(row[ccs_col]).strip()
        if not raw or not ccs_code:
            continue
        ccs_map[raw] = ccs_code
        if desc_col and pd_notna(row.get(desc_col)):
            ccs_desc[ccs_code] = str(row[desc_col]).strip()
    return ccs_map, ccs_desc


def pd_notna(value) -> bool:
    return value is not None and str(value).lower() != "nan"


def normalize_ndc(value: str) -> str:
    return str(value).strip().replace("-", "").replace(" ", "")


def _clean_hcup_token(value: str) -> str:
    return value.strip().strip("'").strip().strip('"').strip()


def _parse_hcup_ccs_file(
    path: Path,
    code_key: str,
    ccs_key: str,
    desc_key: str,
) -> tuple[dict[str, str], dict[str, str]]:
    code_to_ccs: dict[str, str] = {}
    ccs_to_desc: dict[str, str] = {}
    with path.open("r", encoding="latin1", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        header: list[str] | None = None
        col_idx: dict[str, int] = {}
        for row in reader:
            cleaned = [_clean_hcup_token(x) for x in row]
            if not cleaned:
                continue
            if header is None:
                maybe_header = {c.upper(): i for i, c in enumerate(cleaned)}
                if code_key in maybe_header and ccs_key in maybe_header:
                    header = cleaned
                    col_idx["code"] = maybe_header[code_key]
                    col_idx["ccs"] = maybe_header[ccs_key]
                    col_idx["desc"] = maybe_header.get(desc_key, -1)
                continue
            code = cleaned[col_idx["code"]] if col_idx["code"] < len(cleaned) else ""
            ccs = cleaned[col_idx["ccs"]] if col_idx["ccs"] < len(cleaned) else ""
            desc = cleaned[col_idx["desc"]] if col_idx["desc"] >= 0 and col_idx["desc"] < len(cleaned) else ""
            code_norm = normalize_code(code)
            if not code_norm or not ccs:
                continue
            code_to_ccs[code_norm] = ccs
            if desc and ccs not in ccs_to_desc:
                ccs_to_desc[ccs] = desc
    return code_to_ccs, ccs_to_desc


def create_ccs_maps(con: duckdb.DuckDBPyConnection, mappings_dir: Path) -> None:
    ccs10_zip = ensure_download(CCS10_URL, mappings_dir / "ccs_pr_icd10pcs_2019_1.zip")
    ccs9_zip = ensure_download(CCS9_URL, mappings_dir / "Single_Level_CCS_2015.zip")
    extracted = extract_zip(ccs10_zip, mappings_dir / "ccs_procedure_downloads")
    extracted.extend(extract_zip(ccs9_zip, mappings_dir / "ccs_procedure_downloads"))

    icd10_file = next((p for p in extracted if p.name.lower() == "ccs_pr_icd10pcs_2019_1.csv"), None)
    icd9_file = next((p for p in extracted if "$prref" in p.name.lower()), None)
    prlabel_file = next((p for p in extracted if "prlabel" in p.name.lower()), None)

    if icd10_file is None or icd9_file is None:
        raise FileNotFoundError("Missing required CCS procedure files after extraction.")

    proc10_map, ccs_desc_10 = _parse_hcup_ccs_file(
        icd10_file,
        code_key="ICD-10-PCS CODE",
        ccs_key="CCS CATEGORY",
        desc_key="CCS CATEGORY DESCRIPTION",
    )
    proc9_map, ccs_desc_9 = _parse_hcup_ccs_file(
        icd9_file,
        code_key="ICD-9-CM CODE",
        ccs_key="CCS CATEGORY",
        desc_key="CCS CATEGORY DESCRIPTION",
    )

    ccs_desc = {}
    ccs_desc.update(ccs_desc_10)
    ccs_desc.update(ccs_desc_9)
    if prlabel_file is not None:
        labels_map, labels_desc = _parse_hcup_ccs_file(
            prlabel_file,
            code_key="CCS PROCEDURE CATEGORIES",
            ccs_key="CCS PROCEDURE CATEGORIES",
            desc_key="CCS PROCEDURE CATEGORIES LABELS",
        )
        del labels_map
        ccs_desc.update(labels_desc)

    _dict_to_duckdb_table(con, proc9_map, "ccs_proc9", ("icd_code_norm", "ccs_code"))
    _dict_to_duckdb_table(con, proc10_map, "ccs_proc10", ("icd_code_norm", "ccs_code"))
    _dict_to_duckdb_table(con, ccs_desc, "ccs_desc", ("ccs_code", "description"))

    LOGGER.info("Loaded CCS maps: proc9=%d, proc10=%d, descriptions=%d", len(proc9_map), len(proc10_map), len(ccs_desc))


def rollup_medications_with_cache(
    con: duckdb.DuckDBPyConnection,
    mappings_dir: Path,
    force_rxnorm: bool = False,
) -> None:
    cache_path = mappings_dir / "ndc_rxnorm_cache.json"
    name_cache_path = mappings_dir / "rxcui_name_cache.json"

    if not force_rxnorm and cache_path.exists() and name_cache_path.exists():
        cache_warm = load_json(cache_path)
        names_warm = load_json(name_cache_path)
        if len(cache_warm) > 1000 and len(names_warm) > 1000:
            LOGGER.info(
                "Reusing cached RxNorm mappings (%d NDC->RXCUI, %d RxCUI names); "
                "skipping RxNorm zip download/parsing.",
                len(cache_warm),
                len(names_warm),
            )
            _dict_to_duckdb_table(con, cache_warm, "ndc_rxnorm_map", ("ndc", "rxcui"))
            return

    cache = load_json(cache_path)

    # Bulk load NDC->RXCUI from official RxNorm release (RXNSAT.RRF).
    rxnorm_zip = ensure_download(RXNORM_FULL_URL, mappings_dir / "RxNorm_full_prescribe_04062026.zip")
    extract_root = mappings_dir / "rxnorm_extracted"
    extract_root.mkdir(parents=True, exist_ok=True)
    bulk_pairs: dict[str, str] = {}
    rxcui_names: dict[str, str] = {}

    with zipfile.ZipFile(rxnorm_zip, "r") as zf:
        names = zf.namelist()
        rxnsat_member = next((n for n in names if n.upper().endswith("RXNSAT.RRF")), None)
        rxnconso_member = next((n for n in names if n.upper().endswith("RXNCONSO.RRF")), None)
        if rxnsat_member is None:
            raise FileNotFoundError("RXNSAT.RRF not found in RxNorm full release zip.")
        rxnsat_path = extract_root / "RXNSAT.RRF"
        with zf.open(rxnsat_member) as src, rxnsat_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        rxnconso_path: Path | None
        if rxnconso_member:
            rxnconso_path = extract_root / "RXNCONSO.RRF"
            with zf.open(rxnconso_member) as src, rxnconso_path.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        else:
            rxnconso_path = None
            LOGGER.warning("RXNCONSO.RRF not found in RxNorm full release zip; RXN names may fallback.")

    progress_interval = 5_000_000
    LOGGER.info("Parsing RXNSAT.RRF from disk: %s", rxnsat_path)
    t0 = time.perf_counter()
    line_count = 0
    with rxnsat_path.open("rb") as f:
        for raw in f:
            line_count += 1
            if line_count % progress_interval == 0:
                LOGGER.info(
                    "RXNSAT.RRF progress: lines=%d, ndc_mappings=%d, elapsed=%.1fs",
                    line_count,
                    len(bulk_pairs),
                    time.perf_counter() - t0,
                )
            line = raw.decode("utf-8", errors="ignore")
            parts = line.split("|")
            if len(parts) < 11:
                continue
            rxcui = parts[0].strip()
            atn = parts[8].strip()
            atv = parts[10].strip()
            if atn != "NDC" or not rxcui or not atv:
                continue
            ndc = normalize_ndc(atv)
            if ndc:
                bulk_pairs[ndc] = rxcui
    LOGGER.info(
        "RXNSAT.RRF complete: lines=%d, ndc_mappings=%d, elapsed=%.1fs",
        line_count,
        len(bulk_pairs),
        time.perf_counter() - t0,
    )

    if rxnconso_path is not None:
        LOGGER.info("Parsing RXNCONSO.RRF from disk: %s", rxnconso_path)
        t1 = time.perf_counter()
        line_count_c = 0
        with rxnconso_path.open("rb") as f:
            for raw in f:
                line_count_c += 1
                if line_count_c % progress_interval == 0:
                    LOGGER.info(
                        "RXNCONSO.RRF progress: lines=%d, name_mappings=%d, elapsed=%.1fs",
                        line_count_c,
                        len(rxcui_names),
                        time.perf_counter() - t1,
                    )
                line = raw.decode("utf-8", errors="ignore")
                parts = line.split("|")
                if len(parts) < 15:
                    continue
                rxcui = parts[0].strip()
                tty = parts[12].strip()
                name = parts[14].strip()
                if not rxcui or not name:
                    continue
                if tty == "IN":
                    rxcui_names[rxcui] = name
                elif tty in ("PIN", "BN") and rxcui not in rxcui_names:
                    rxcui_names[rxcui] = name
                elif rxcui not in rxcui_names:
                    # Backfill with any available term if IN/PIN/BN absent.
                    rxcui_names[rxcui] = name
        LOGGER.info(
            "RXNCONSO.RRF complete: lines=%d, name_mappings=%d, elapsed=%.1fs",
            line_count_c,
            len(rxcui_names),
            time.perf_counter() - t1,
        )
    save_json_fast(name_cache_path, rxcui_names)
    LOGGER.info("Saved %d RxCUI->name mappings from RXNCONSO.RRF", len(rxcui_names))
    LOGGER.info("Loaded %d NDC->RXCUI pairs from RXNSAT.RRF", len(bulk_pairs))
    cache.update(bulk_pairs)

    _dict_to_duckdb_table(con, cache, "ndc_rxnorm_map", ("ndc", "rxcui"))
    save_json_fast(cache_path, cache)
    LOGGER.info("NDC->RxNorm cache size: %d (bulk only, no API calls)", len(cache))


def build_rollup(con: duckdb.DuckDBPyConnection, rolled_output_path: Path) -> None:
    LOGGER.info("Starting main rollup query (CREATE OR REPLACE TABLE patient_events_rolled)...")
    rollup_t0 = time.perf_counter()
    con.execute(
        """
        CREATE OR REPLACE TABLE patient_events_rolled AS
        SELECT
            e.subject_id,
            e.hadm_id,
            e.event_time,
            CASE
                WHEN e.code_type = 'diagnosis' AND e.code_id LIKE 'ICD9_%' AND p9.phecode IS NOT NULL THEN 'PHE_' || p9.phecode
                WHEN e.code_type = 'diagnosis' AND e.code_id LIKE 'ICD10_%' AND p10.phecode IS NOT NULL THEN 'PHE_' || p10.phecode
                WHEN e.code_type = 'procedure' AND e.code_id LIKE 'PROC9_%' AND c9.ccs_code IS NOT NULL THEN 'CCS_' || c9.ccs_code
                WHEN e.code_type = 'procedure' AND e.code_id LIKE 'PROC10_%' AND c10.ccs_code IS NOT NULL THEN 'CCS_' || c10.ccs_code
                WHEN e.code_type = 'hcpcs' AND e.code_id LIKE 'HCPCS_%' AND c_hcpcs.ccs_code IS NOT NULL THEN 'CCS_' || c_hcpcs.ccs_code
                WHEN e.code_type = 'medication' AND e.code_id LIKE 'NDC_%' AND rx.rxcui IS NOT NULL THEN 'RXN_' || rx.rxcui
                ELSE e.code_id
            END AS code_id,
            CASE
                WHEN e.code_type = 'diagnosis'
                     AND (
                         (e.code_id LIKE 'ICD9_%' AND p9.phecode IS NULL)
                         OR (e.code_id LIKE 'ICD10_%' AND p10.phecode IS NULL)
                     ) THEN 'diagnosis_unmapped'
                WHEN e.code_type = 'procedure'
                     AND (
                         (e.code_id LIKE 'PROC9_%' AND c9.ccs_code IS NULL)
                         OR (e.code_id LIKE 'PROC10_%' AND c10.ccs_code IS NULL)
                     ) THEN 'procedure_unmapped'
                WHEN e.code_type = 'hcpcs'
                     AND (e.code_id LIKE 'HCPCS_%' AND c_hcpcs.ccs_code IS NULL) THEN 'hcpcs_unmapped'
                WHEN e.code_type = 'medication'
                     AND (e.code_id LIKE 'NDC_%' AND rx.rxcui IS NULL) THEN 'medication_unmapped'
                ELSE e.code_type
            END AS code_type,
            e.timestamp_days,
            e.log_delta_t,
            e.age_at_event_days,
            e.sex,
            e.race
        FROM patient_events e
        LEFT JOIN phe_icd9 p9
            ON e.code_type = 'diagnosis'
           AND e.code_id LIKE 'ICD9_%'
           AND REPLACE(UPPER(TRIM(SUBSTR(e.code_id, 6))), '.', '') = p9.icd_code_norm
        LEFT JOIN phe_icd10 p10
            ON e.code_type = 'diagnosis'
           AND e.code_id LIKE 'ICD10_%'
           AND REPLACE(UPPER(TRIM(SUBSTR(e.code_id, 7))), '.', '') = p10.icd_code_norm
        LEFT JOIN ccs_proc9 c9
            ON e.code_type = 'procedure'
           AND e.code_id LIKE 'PROC9_%'
           AND REPLACE(UPPER(TRIM(SUBSTR(e.code_id, 7))), '.', '') = c9.icd_code_norm
        LEFT JOIN ccs_proc10 c10
            ON e.code_type = 'procedure'
           AND e.code_id LIKE 'PROC10_%'
           AND REPLACE(UPPER(TRIM(SUBSTR(e.code_id, 8))), '.', '') = c10.icd_code_norm
        LEFT JOIN ccs_proc10 c_hcpcs
            ON e.code_type = 'hcpcs'
           AND e.code_id LIKE 'HCPCS_%'
           AND REPLACE(UPPER(TRIM(SUBSTR(e.code_id, 7))), '.', '') = c_hcpcs.icd_code_norm
        LEFT JOIN ndc_rxnorm_map rx
            ON e.code_type = 'medication'
           AND e.code_id LIKE 'NDC_%'
           AND REPLACE(REPLACE(TRIM(SUBSTR(e.code_id, 5)), '-', ''), ' ', '') = rx.ndc
        """
    )
    LOGGER.info(
        "Main rollup query finished in %.1fs (patient_events_rolled materialized).",
        time.perf_counter() - rollup_t0,
    )

    escaped_output = str(rolled_output_path).replace("'", "''")
    LOGGER.info("Writing rolled parquet via COPY ... TO '%s'", rolled_output_path)
    copy_t0 = time.perf_counter()
    con.execute(
        f"""
        COPY (
            SELECT
                subject_id,
                hadm_id,
                event_time,
                code_id,
                code_type,
                timestamp_days,
                log_delta_t,
                age_at_event_days,
                sex,
                race
            FROM patient_events_rolled
            ORDER BY subject_id, event_time, code_id
        ) TO '{escaped_output}' (FORMAT PARQUET)
        """
    )
    LOGGER.info("COPY to parquet finished in %.1fs.", time.perf_counter() - copy_t0)


def print_rollup_stats(con: duckdb.DuckDBPyConnection) -> None:
    before = int(con.execute("SELECT COUNT(DISTINCT code_id) FROM patient_events").fetchone()[0])
    after = int(con.execute("SELECT COUNT(DISTINCT code_id) FROM patient_events_rolled").fetchone()[0])
    print(f"Unique codes before rollup: {before:,}")
    print(f"Unique codes after rollup: {after:,}")

    print("Per code_type mapping stats:")
    originals = {
        k: int(v)
        for k, v in con.execute(
            """
            SELECT code_type, COUNT(*) AS cnt
            FROM patient_events
            GROUP BY code_type
            """
        ).fetchall()
    }
    rolled = {
        k: int(v)
        for k, v in con.execute(
            """
            SELECT code_type, COUNT(*) AS cnt
            FROM patient_events_rolled
            GROUP BY code_type
            """
        ).fetchall()
    }
    for code_type in sorted(originals):
        total = originals.get(code_type, 0)
        if code_type in ("lab", "chart", "drg", "input", "output", "icu_procedure"):
            mapped = total
            unmapped = 0
        else:
            mapped = rolled.get(code_type, 0)
            unmapped = rolled.get(f"{code_type}_unmapped", 0)
        rate = (float(mapped) / float(total) * 100.0) if total else 0.0
        print(
            f"  - {code_type}: total={total:,}, mapped={mapped:,}, "
            f"unmapped={unmapped:,}, mapping_rate={rate:.2f}%"
        )

    print("Top 10 most frequent unmapped codes:")
    top_unmapped = con.execute(
        """
        SELECT code_id, code_type, COUNT(*) AS cnt
        FROM patient_events_rolled
        WHERE code_type LIKE '%_unmapped'
        GROUP BY code_id, code_type
        ORDER BY cnt DESC
        LIMIT 10
        """
    ).fetchall()
    for code_id, code_type, cnt in top_unmapped:
        print(f"  - {code_id} [{code_type}]: {int(cnt):,}")


def build_description_dict(
    con: duckdb.DuckDBPyConnection,
    mimic_root: Path,
    mappings_dir: Path,
) -> dict[str, str]:
    LOGGER.info("build_description_dict: loading reference tables from MIMIC dictionaries...")
    desc_t0 = time.perf_counter()
    _execute_csv_sql(
        con,
        """
        CREATE OR REPLACE TABLE d_labitems AS
        SELECT
            CAST(TRY_CAST(itemid AS BIGINT) AS VARCHAR) AS itemid,
            CAST(label AS VARCHAR) AS label
        FROM {source}
        WHERE TRY_CAST(itemid AS BIGINT) IS NOT NULL
        """,
        mimic_root / "hosp" / "d_labitems.csv.gz",
    )
    LOGGER.info("build_description_dict: d_labitems loaded (elapsed %.1fs).", time.perf_counter() - desc_t0)
    _execute_csv_sql(
        con,
        """
        CREATE OR REPLACE TABLE d_items AS
        SELECT
            CAST(TRY_CAST(itemid AS BIGINT) AS VARCHAR) AS itemid,
            CAST(label AS VARCHAR) AS label,
            CAST(category AS VARCHAR) AS category
        FROM {source}
        WHERE TRY_CAST(itemid AS BIGINT) IS NOT NULL
        """,
        mimic_root / "icu" / "d_items.csv.gz",
    )
    LOGGER.info("build_description_dict: d_items loaded (elapsed %.1fs).", time.perf_counter() - desc_t0)
    _execute_csv_sql(
        con,
        """
        CREATE OR REPLACE TABLE drg_desc AS
        SELECT drg_code, description
        FROM (
            SELECT
                CAST(drg_code AS VARCHAR) AS drg_code,
                CAST(description AS VARCHAR) AS description,
                COUNT(*) AS cnt,
                ROW_NUMBER() OVER (
                    PARTITION BY CAST(drg_code AS VARCHAR)
                    ORDER BY COUNT(*) DESC
                ) AS rn
            FROM {source}
            WHERE drg_code IS NOT NULL AND description IS NOT NULL
            GROUP BY CAST(drg_code AS VARCHAR), CAST(description AS VARCHAR)
        ) x
        WHERE rn = 1
        """,
        mimic_root / "hosp" / "drgcodes.csv.gz",
    )
    LOGGER.info("build_description_dict: drg_desc loaded (elapsed %.1fs).", time.perf_counter() - desc_t0)
    _execute_csv_sql(
        con,
        """
        CREATE OR REPLACE TABLE d_icd_diagnoses AS
        SELECT
            REPLACE(UPPER(TRIM(CAST(icd_code AS VARCHAR))), '.', '') AS icd_code_norm,
            TRY_CAST(icd_version AS BIGINT) AS icd_version,
            CAST(long_title AS VARCHAR) AS long_title
        FROM {source}
        WHERE icd_code IS NOT NULL
        """,
        mimic_root / "hosp" / "d_icd_diagnoses.csv.gz",
    )
    LOGGER.info("build_description_dict: d_icd_diagnoses loaded (elapsed %.1fs).", time.perf_counter() - desc_t0)
    _execute_csv_sql(
        con,
        """
        CREATE OR REPLACE TABLE d_icd_procedures AS
        SELECT
            REPLACE(UPPER(TRIM(CAST(icd_code AS VARCHAR))), '.', '') AS icd_code_norm,
            TRY_CAST(icd_version AS BIGINT) AS icd_version,
            CAST(long_title AS VARCHAR) AS long_title
        FROM {source}
        WHERE icd_code IS NOT NULL
        """,
        mimic_root / "hosp" / "d_icd_procedures.csv.gz",
    )
    LOGGER.info("build_description_dict: d_icd_procedures loaded (elapsed %.1fs).", time.perf_counter() - desc_t0)
    d_hcpcs_path = mimic_root / "hosp" / "d_hcpcs.csv.gz"
    if d_hcpcs_path.exists():
        _execute_csv_sql(
            con,
            """
            CREATE OR REPLACE TABLE d_hcpcs AS
            SELECT
                TRIM(CAST(code AS VARCHAR)) AS hcpcs_code,
                CAST(short_description AS VARCHAR) AS short_description
            FROM {source}
            WHERE code IS NOT NULL
            """,
            d_hcpcs_path,
        )
        LOGGER.info("build_description_dict: d_hcpcs loaded (elapsed %.1fs).", time.perf_counter() - desc_t0)
    else:
        LOGGER.warning(
            "build_description_dict: d_hcpcs.csv.gz not found at %s; HCPCS descriptions will use fallback.",
            d_hcpcs_path,
        )
        con.execute(
            """
            CREATE OR REPLACE TABLE d_hcpcs AS
            SELECT
                CAST(NULL AS VARCHAR) AS hcpcs_code,
                CAST(NULL AS VARCHAR) AS short_description
            WHERE 1 = 0
            """
        )

    # RxNorm name cache.
    name_cache_path = mappings_dir / "rxcui_name_cache.json"
    name_cache = load_json(name_cache_path)
    _dict_to_duckdb_table(con, name_cache, "rxcui_name_map", ("rxcui", "name"))
    LOGGER.info(
        "build_description_dict: rxcui_name_map registered (%d names, elapsed %.1fs).",
        len(name_cache),
        time.perf_counter() - desc_t0,
    )

    desc_sql = """
        WITH codes AS (
            SELECT DISTINCT code_id, code_type
            FROM patient_events_rolled
        )
        SELECT
            c.code_id,
            COALESCE(
                CASE
                    WHEN c.code_id LIKE 'PHE_%' THEN p.phenotype || ' (' || p.category || ')'
                    WHEN c.code_id LIKE 'CCS_%' THEN cd.description || ' (procedure)'
                    WHEN c.code_id LIKE 'RXN_%' THEN COALESCE(rn.name || ' (medication)', 'RxNorm ' || SUBSTR(c.code_id, 5))
                    WHEN c.code_id LIKE 'LAB_%' THEN dl.label || ' (laboratory measurement)'
                    WHEN c.code_id LIKE 'CHART_%' THEN di.label || ' (' || di.category || ')'
                    WHEN c.code_id LIKE 'DRG_%' THEN dd.description || ' (diagnosis related group)'
                    WHEN c.code_id LIKE 'INPUT_%' THEN di_input.label || ' (ICU input)'
                    WHEN c.code_id LIKE 'OUTPUT_%' THEN di_output.label || ' (ICU output)'
                    WHEN c.code_id LIKE 'ICUPROC_%' THEN di_icuproc.label || ' (ICU procedure)'
                    WHEN c.code_id LIKE 'HCPCS_%' THEN COALESCE(dh.short_description, 'HCPCS ' || SUBSTR(c.code_id, 7))
                    WHEN c.code_type = 'diagnosis_unmapped' AND c.code_id LIKE 'ICD9_%' THEN did9.long_title
                    WHEN c.code_type = 'diagnosis_unmapped' AND c.code_id LIKE 'ICD10_%' THEN did10.long_title
                    WHEN c.code_id LIKE 'PROC9_%' THEN dip9.long_title
                    WHEN c.code_id LIKE 'PROC10_%' THEN dip10.long_title
                    WHEN c.code_id LIKE 'NDC_%' THEN 'NDC code ' || SUBSTR(c.code_id, 5)
                    ELSE NULL
                END,
                c.code_id
            ) AS description
        FROM codes c
        LEFT JOIN phe_defs p
            ON c.code_id LIKE 'PHE_%'
           AND SUBSTR(c.code_id, 5) = p.phecode
        LEFT JOIN ccs_desc cd
            ON c.code_id LIKE 'CCS_%'
           AND SUBSTR(c.code_id, 5) = cd.ccs_code
        LEFT JOIN rxcui_name_map rn
            ON c.code_id LIKE 'RXN_%'
           AND SUBSTR(c.code_id, 5) = rn.rxcui
        LEFT JOIN d_labitems dl
            ON c.code_id LIKE 'LAB_%'
           AND SUBSTR(c.code_id, 5) = dl.itemid
        LEFT JOIN d_items di
            ON c.code_id LIKE 'CHART_%'
           AND SUBSTR(c.code_id, 7) = di.itemid
        LEFT JOIN d_items di_input
            ON c.code_id LIKE 'INPUT_%'
           AND SUBSTR(c.code_id, 7) = di_input.itemid
        LEFT JOIN d_items di_output
            ON c.code_id LIKE 'OUTPUT_%'
           AND SUBSTR(c.code_id, 8) = di_output.itemid
        LEFT JOIN d_items di_icuproc
            ON c.code_id LIKE 'ICUPROC_%'
           AND SUBSTR(c.code_id, 9) = di_icuproc.itemid
        LEFT JOIN drg_desc dd
            ON c.code_id LIKE 'DRG_%'
           AND SUBSTR(c.code_id, 5) = dd.drg_code
        LEFT JOIN d_hcpcs dh
            ON c.code_id LIKE 'HCPCS_%'
           AND SUBSTR(c.code_id, 7) = dh.hcpcs_code
        LEFT JOIN d_icd_diagnoses did9
            ON c.code_type = 'diagnosis_unmapped'
           AND c.code_id LIKE 'ICD9_%'
           AND REPLACE(UPPER(TRIM(SUBSTR(c.code_id, 6))), '.', '') = did9.icd_code_norm
           AND did9.icd_version = 9
        LEFT JOIN d_icd_diagnoses did10
            ON c.code_type = 'diagnosis_unmapped'
           AND c.code_id LIKE 'ICD10_%'
           AND REPLACE(UPPER(TRIM(SUBSTR(c.code_id, 7))), '.', '') = did10.icd_code_norm
           AND did10.icd_version = 10
        LEFT JOIN d_icd_procedures dip9
            ON c.code_id LIKE 'PROC9_%'
           AND REPLACE(UPPER(TRIM(SUBSTR(c.code_id, 7))), '.', '') = dip9.icd_code_norm
           AND dip9.icd_version = 9
        LEFT JOIN d_icd_procedures dip10
            ON c.code_id LIKE 'PROC10_%'
           AND REPLACE(UPPER(TRIM(SUBSTR(c.code_id, 8))), '.', '') = dip10.icd_code_norm
           AND dip10.icd_version = 10
        """
    LOGGER.info("build_description_dict: executing description join query...")
    query_t0 = time.perf_counter()
    result = con.execute(desc_sql)
    descriptions: dict[str, str] = {}
    batch_size = 500_000
    progress_interval = 1_000_000
    rows_fetched = 0
    next_log_at = progress_interval
    while True:
        chunk = result.fetchmany(batch_size)
        if not chunk:
            break
        for code, desc_val in chunk:
            descriptions[str(code)] = str(desc_val)
        rows_fetched += len(chunk)
        if rows_fetched >= next_log_at:
            LOGGER.info(
                "build_description_dict: fetched %d description rows, elapsed %.1fs",
                rows_fetched,
                time.perf_counter() - query_t0,
            )
            next_log_at += progress_interval
    LOGGER.info(
        "build_description_dict: complete (%d codes, query+fetch %.1fs, total %.1fs).",
        len(descriptions),
        time.perf_counter() - query_t0,
        time.perf_counter() - desc_t0,
    )
    return descriptions


def print_description_validation(con: duckdb.DuckDBPyConnection, descriptions: dict[str, str]) -> None:
    all_codes = [str(r[0]) for r in con.execute("SELECT DISTINCT code_id FROM patient_events_rolled").fetchall()]
    missing = [c for c in all_codes if c not in descriptions or not descriptions[c]]
    if missing:
        print("Missing descriptions (up to 20):")
        for code in missing[:20]:
            print(f"  - {code}")
    else:
        print("Missing descriptions (up to 20): none")

    if missing:
        raise AssertionError(f"Missing descriptions for {len(missing)} code_ids")

    print(f"Total codes with descriptions: {len(descriptions):,}")
    coverage_rows = con.execute(
        """
        WITH codes AS (
            SELECT DISTINCT code_id
            FROM patient_events_rolled
        )
        SELECT
            CASE
                WHEN code_id LIKE 'PHE_%' THEN 'PHE'
                WHEN code_id LIKE 'CCS_%' THEN 'CCS'
                WHEN code_id LIKE 'RXN_%' THEN 'RXN'
                WHEN code_id LIKE 'LAB_%' THEN 'LAB'
                WHEN code_id LIKE 'CHART_%' THEN 'CHART'
                WHEN code_id LIKE 'DRG_%' THEN 'DRG'
                WHEN code_id LIKE 'ICD9_%' THEN 'ICD9'
                WHEN code_id LIKE 'ICD10_%' THEN 'ICD10'
                WHEN code_id LIKE 'PROC9_%' THEN 'PROC9'
                WHEN code_id LIKE 'PROC10_%' THEN 'PROC10'
                WHEN code_id LIKE 'NDC_%' THEN 'NDC'
                ELSE 'OTHER'
            END AS prefix_type,
            COUNT(*) AS cnt
        FROM codes
        GROUP BY prefix_type
        ORDER BY cnt DESC
        """
    ).fetchall()
    print("Coverage by prefix type:")
    for prefix, cnt in coverage_rows:
        print(f"  - {prefix}: {int(cnt):,}")


def main() -> int:
    setup_logging()
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    processed_dir = repo_root / "data" / "processed"
    mappings_dir = processed_dir / "mappings"
    mappings_dir.mkdir(parents=True, exist_ok=True)

    suffix = "test" if args.test_mode else "full"
    input_parquet = processed_dir / f"patient_events_{suffix}.parquet"
    rolled_parquet = processed_dir / f"patient_events_rolled_{suffix}.parquet"
    desc_json_path = processed_dir / "code_descriptions.json"

    if not input_parquet.exists():
        raise FileNotFoundError(f"Missing input parquet: {input_parquet}")

    mimic_root = resolve_mimic_root(repo_root)

    LOGGER.info("Checking existing NDC/RxNorm assets in external pipeline.")
    ndc_rx_files = find_existing_ndc_rxnorm_files()
    for fp in ndc_rx_files[:10]:
        LOGGER.info("Found mapping candidate: %s", fp)

    # con = duckdb.connect() # replacing with the following:
    db_path = processed_dir / "rollup_and_describe.duckdb"
    wal_path = processed_dir / "rollup_and_describe.duckdb.wal"
    if db_path.exists():
        db_path.unlink()
    if wal_path.exists():
        wal_path.unlink()

    con = duckdb.connect(str(db_path))
    # replacement ends here
    
    try:
        con.execute("PRAGMA memory_limit='24GB'")
        duckdb_tmp = (processed_dir / "duckdb_tmp").resolve()
        duckdb_tmp.mkdir(parents=True, exist_ok=True)
        escaped_tmp = str(duckdb_tmp).replace("'", "''")
        con.execute(f"PRAGMA temp_directory='{escaped_tmp}'")
        con.execute(f"PRAGMA threads={max(1, (os.cpu_count() or 8) - 2)}")

        con.execute(
            f"""
            CREATE OR REPLACE TABLE patient_events AS
            SELECT *
            FROM read_parquet('{str(input_parquet).replace("'", "''")}')
            """
        )
        create_phecode_maps(con, mappings_dir)
        create_ccs_maps(con, mappings_dir)
        rollup_medications_with_cache(con, mappings_dir, force_rxnorm=args.force_rxnorm)
        build_rollup(con, rolled_parquet)
        print_rollup_stats(con)

        descriptions = build_description_dict(con, mimic_root, mappings_dir)
        save_json(desc_json_path, descriptions)
        print_description_validation(con, descriptions)

        LOGGER.info("Wrote rolled parquet: %s", rolled_parquet)
        LOGGER.info("Wrote code descriptions: %s", desc_json_path)
        return 0
    finally:
        con.close()


if __name__ == "__main__":
    raise SystemExit(main())