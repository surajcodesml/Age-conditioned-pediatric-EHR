import numpy as np
import pandas as pd
from pathlib import Path
import json

base = Path("outputs/data/seed20260922/controlled")
out_dir = base / "S2_multi_horizon"
out_dir.mkdir(exist_ok=True, parents=True)

horizons = [30, 90, 180, 365]

dfs = []
npzs = []
specs = []

for h in horizons:
    d = base / f"S2_h{h}"
    dfs.append(pd.read_parquet(d / "examples.parquet"))
    npzs.append(np.load(d / "labels.npz"))
    specs.append(json.loads((d / "target_specs.json").read_text()))

df_out = dfs[0].copy()
Y_all = []
for i, row in enumerate(df_out.itertuples(index=False)):
    labels = []
    for df in dfs:
        labels.extend(df.at[i, "labels"])
    Y_all.append(labels)
    df_out.at[i, "labels"] = labels

df_out.to_parquet(out_dir / "examples.parquet", index=False)

Y = np.concatenate([n["Y"] for n in npzs], axis=1)
logits = np.concatenate([n["logits"] for n in npzs], axis=1)
probs = np.concatenate([n["probs"] for n in npzs], axis=1)
np.savez_compressed(
    out_dir / "labels.npz",
    Y=Y,
    logits=logits,
    probs=probs,
    ages=npzs[0]["ages"],
    true_lambda=npzs[0]["true_lambda"]
)

# merge specs
specs_out = []
for h, sp_list in zip(horizons, specs):
    for sp in sp_list:
        sp_new = dict(sp)
        sp_new["name"] = f"{sp['name']}_h{h}"
        specs_out.append(sp_new)

(out_dir / "target_specs.json").write_text(json.dumps(specs_out, indent=2))
(out_dir / "meta.json").write_text((base / "S2_h30" / "meta.json").read_text())
(out_dir / "splits.json").write_text((base / "S2_h30" / "splits.json").read_text())
print("Created S2_multi_horizon")
