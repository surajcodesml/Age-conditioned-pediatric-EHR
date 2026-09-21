# MIMIC ↔ NCH vocabulary compatibility

NCH codes are normalized with the Stage-1 rule `strip/upper/remove-dots` and
rolled through the **same** PheWAS ICD→PheCode maps used for MIMIC.
The frozen `data/processed/code_vocab.json` (`|V|=30635`) is never modified.

PSG waveforms are **not** part of the EHR vocabulary. Sleep studies are index times.

## Coverage by event family

| family | NCH rows | unique raw | unique norm | vocab matches | unique % | event % | OOV events | OOV % | patients with OOV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| diagnosis | 1,513,853 | 11,472 | 11,467 | 5,984 | 52.2 | 91.2 | 133,927 | 8.8 | 3,088 (84.1%) |
| drg | 11,594 | 663 | 663 | 0 | 0.0 | 0.0 | 11,594 | 100.0 | 3,648 (100.0%) |
| measurement | 332,569 | 3 | 0 | 0 | 0.0 | 0.0 | 332,569 | 100.0 | 0 (0.0%) |
| medication | 3,035,986 | 2,616 | 2,616 | 949 | 36.3 | 31.9 | 2,068,441 | 68.1 | 3,366 (97.1%) |
| procedure | 293,789 | 3,684 | 3,680 | 204 | 5.5 | 14.7 | 250,544 | 85.3 | 3,665 (100.0%) |

Codes that matched only after formatting normalization: **10132** unique rows in `code_mapping.csv`.

## OOV cause breakdown (unique-code table, event-weighted)

| family | cause | events | unique codes |
|---|---|---:|---:|
| diagnosis | code_version_mismatch | 64,455 | 225 |
| diagnosis | unknown_unmappable | 56,541 | 1,520 |
| diagnosis | pediatric_absent | 6,471 | 135 |
| diagnosis | redacted | 6,460 | 2 |
| drg | unknown_unmappable | 11,594 | 663 |
| medication | unknown_unmappable | 1,136,847 | 1 |
| medication | pediatric_absent | 931,594 | 1,667 |
| procedure | unknown_unmappable | 249,857 | 954 |
| procedure | nch_local | 687 | 1 |

Cause legend:

1. `formatting_mismatch` — would match after case/dot/zero normalization (these are mapped, not left OOV).
2. `coding_system_mismatch` — NCH system is not one Stage-1 rolled (e.g. local procedure IDs).
3. `code_version_mismatch` — ICD-9 vs ICD-10 leftover not in the frozen vocab.
4. `nch_local` — NCH internal identifiers (`PROC_ID_NCH`, `SHX*` surgical history).
5. `pediatric_absent` — maps to a well-formed Stage-1 *namespace* token (PHE_/RXN_) that adult MIMIC never kept.
6. `redacted` — placeholder / missing clinical code.
7. `unknown_unmappable` — no justified map onto the frozen vocabulary.

## Most frequent matched concepts

- `RXN_630208` (medication): 68,453 events — ALBUTEROL SULFATE 2.5 mg in 3 mL RESPIRATORY (INHALATION) SOLUTION (medication)
- `PHE_315.2` (diagnosis): 47,412 events — Speech and language disorder (mental disorders)
- `RXN_1807627` (medication): 43,473 events — SODIUM CHLORIDE 900 mg in 100 mL INTRAVENOUS INJECTION, SOLUTION [Sodium Chloride]_#1 (medication)
- `PHE_758.1` (diagnosis): 37,136 events — Chromosomal anomalies (congenital anomalies)
- `RXN_197803` (medication): 35,374 events — IBUPROFEN 100 mg in 5 mL ORAL SUSPENSION (medication)
- `PHE_343` (diagnosis): 35,095 events — Infantile cerebral palsy (neurological)
- `RXN_847630` (medication): 30,218 events — SODIUM CHLORIDE 600 mg in 100 mL / SODIUM LACTATE 310 mg in 100 mL / POTASSIUM CHLORIDE 30 mg in 100 mL / CALCIUM CHLORIDE 20 mg in 100 mL INTRAPERITONEAL INJECTION, SOLUTION (medication)
- `PHE_313.3` (diagnosis): 24,379 events — Autism (mental disorders)
- `PHE_315` (diagnosis): 24,219 events — Develomental delays and disorders (mental disorders)
- `RXN_312515` (medication): 21,889 events — POTASSIUM CHLORIDE 1.5 g in 15 mL ORAL SOLUTION (medication)
- `RXN_403884` (medication): 20,560 events — LEVETIRACETAM 100 mg in 1 mL ORAL SOLUTION [Levetiracetam Levetiracetam] (medication)
- `RXN_1863605` (medication): 19,634 events — DEXTROSE MONOHYDRATE 50 g in 1000 mL / SODIUM CHLORIDE 4.5 g in 1000 mL / POTASSIUM CHLORIDE 1.49 g in 1000 mL INTRAVENOUS INJECTION, SOLUTION [Potassium Chloride in Dextrose and Sodium Chloride]_#2 (medication)
- `PHE_264.9` (diagnosis): 19,617 events — Lack of normal physiological development, unspecified (endocrine/metabolic)
- `RXN_836358` (medication): 18,909 events — IPRATROPIUM BROMIDE 0.5 mg in 2.5 mL RESPIRATORY (INHALATION) SOLUTION (medication)
- `ICD10_R6250` (diagnosis): 18,093 events — Unspecified lack of expected normal physiological development in childhood
- `RXN_197730` (medication): 17,186 events — FUROSEMIDE 10 mg in 1 mL ORAL SYRUP [Furosemide 1%] (medication)
- `ICD10_R0683` (diagnosis): 16,720 events — Snoring
- `RXN_1795250` (medication): 15,809 events — DEXTROSE MONOHYDRATE 5 g in 100 mL / SODIUM CHLORIDE 0.45 g in 100 mL INTRAVENOUS INJECTION, SOLUTION [Dextrose and Sodium Chloride]_#3 (medication)
- `RXN_876193` (medication): 15,733 events — POLYETHYLENE GLYCOL 3350 17 g in 17 g ORAL POWDER, FOR SOLUTION (medication)
- `PHE_465` (diagnosis): 15,226 events — Acute upper respiratory infections of multiple or unspecified sites (respiratory)
- `PHE_1002` (diagnosis): 15,086 events — PHE_1002
- `RXN_705610` (medication): 14,944 events — RxNorm 705610
- `RXN_312997` (medication): 14,799 events — sodium chloride 0.9 % Inhalation Solution (medication)
- `RXN_283077` (medication): 13,969 events — prednisolone 15 MG (as prednisolone sodium phosphate 20.2 MG) per 5 ML Oral Solution (medication)
- `PHE_530.11` (diagnosis): 13,933 events — GERD (digestive)

## Top 50 OOV concepts by event frequency

- `` [medication/unknown_unmappable] 1,136,847 events — TOBRAMYCIN AEROSOL INPATIENT < 160 MG
- `144450` [medication/pediatric_absent] 135,448 events — infant formula, iron/dha/ara
- `801092` [medication/pediatric_absent] 53,474 events — albuterol sulfate
- `727634` [medication/pediatric_absent] 53,366 events — sodium chloride 0.9 % (flush)
- `94640` [procedure/unknown_unmappable] 47,446 events — COOL AEROSOL NEBULIZER PER DAY
- `307668` [medication/pediatric_absent] 46,809 events — acetaminophen
- `97110` [procedure/unknown_unmappable] 30,191 events — PTS THERAPEUTIC EXERCISE/15MIN
- `725145` [medication/pediatric_absent] 26,712 events — petrolatum,white
- `92507` [procedure/unknown_unmappable] 24,662 events — INDIVIDUAL S&H THERAPY, INTERM
- `107129` [medication/pediatric_absent] 21,858 events — water
- `97112` [procedure/unknown_unmappable] 18,298 events — PT NEUROMUSC RE-EDUC/15 MIN
- `97530` [procedure/unknown_unmappable] 18,283 events — OT FUNCTIONAL TRAINING/15 MIN
- `876863` [medication/pediatric_absent] 17,187 events — multivitamin with iron
- `244638` [medication/pediatric_absent] 16,882 events — cod liver oil/zinc oxide
- `846127` [medication/pediatric_absent] 16,266 events — chlorhexidine gluconate
- `V57.1` [diagnosis/code_version_mismatch] 15,733 events — Other physical therapy
- `V57.3` [diagnosis/code_version_mismatch] 14,931 events — Care involving speech-language therapy
- `1362057` [medication/pediatric_absent] 14,518 events — heparin sodium,porcine/PF
- `200317` [medication/pediatric_absent] 13,833 events — fat emulsions
- `416684` [medication/pediatric_absent] 13,219 events — zinc oxide/petrolatum, yellow
- `312447` [medication/pediatric_absent] 12,695 events — piperacillin sodium/tazobactam
- `836343` [medication/pediatric_absent] 12,607 events — ipratropium bromide
- `1360074` [medication/pediatric_absent] 12,295 events — beclomethasone dipropionate
- `1421893` [medication/pediatric_absent] 12,025 events — lidocaine/transparent dressing
- `V20.2` [diagnosis/code_version_mismatch] 11,539 events — Routine infant or child health check
- `Z00.129` [diagnosis/unknown_unmappable] 11,480 events — Encounter for routine child health examination without abnormal findings
- `895994` [medication/pediatric_absent] 11,436 events — fluticasone propionate
- `646456` [medication/pediatric_absent] 11,122 events — nystatin
- `251154` [medication/pediatric_absent] 10,931 events — cholecalciferol (vitamin D3)
- `V57.21` [diagnosis/code_version_mismatch] 10,883 events — Encounter for occupational therapy
- `94660` [procedure/unknown_unmappable] 10,670 events — BIPAP SUBSEQ DAYS
- `1808217` [medication/pediatric_absent] 9,692 events — propofol
- `71010` [procedure/unknown_unmappable] 9,431 events — PORT CHEST SINGLE
- `1307427` [medication/pediatric_absent] 9,081 events — sildenafil citrate
- `70727` [medication/pediatric_absent] 8,800 events — pedi nutrit,iron,lac-free,fibr
- `895999` [medication/pediatric_absent] 8,726 events — fluticasone propionate
- `864761` [medication/pediatric_absent] 8,608 events — methadone HCl
- `686400` [medication/pediatric_absent] 8,134 events — erythromycin ethylsuccinate
- `99283` [procedure/unknown_unmappable] 7,713 events — PR EMERGENCY DEPARTMENT VISIT MODERATE SEVERITY
- `1795344` [medication/pediatric_absent] 7,337 events — dextrose 5 % and 0.9 % NaCl
- `308189` [medication/pediatric_absent] 7,212 events — amoxicillin
- `999961` [medication/pediatric_absent] 6,444 events — glycopyrrolate
- `71045` [procedure/unknown_unmappable] 6,351 events — CHG RADIOLOGIC EXAM CHEST SINGLE VIEW
- `856940` [medication/pediatric_absent] 6,186 events — hydrocodone/acetaminophen
- `310014` [medication/pediatric_absent] 6,142 events — dornase alfa
- `97124` [procedure/unknown_unmappable] 6,027 events — MASSAGE THERAPY PER 15 MINUTES
- `605320` [medication/pediatric_absent] 5,847 events — Saccharomyces boulardii
- `95810` [procedure/unknown_unmappable] 5,666 events — POLYSOMNOGRAPHY, 6 YRS+, 4 OR MORE ADDL PARAM, W/ TECH
- `74000` [procedure/unknown_unmappable] 4,975 events — X-RAY ABDOM SINGLE
- `151029` [medication/pediatric_absent] 4,909 events — mometasone furoate

## Measurements

NCH measurements (BMI, percentiles, vitals) have no counterpart in the frozen MIMIC LAB_/CHART_ itemid vocabulary and are retained only as covariates.

NCH measurement rows: 332569.

## Inventory snapshot

- Patients: 3673
- Sleep studies: 3984
- Patients with >1 study: 273
- Local EDF files downloaded: 21 (RECORDS lists 3984; waveforms unused)
