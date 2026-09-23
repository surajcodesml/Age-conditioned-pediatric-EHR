import os
import json
import duckdb
import torch
import numpy as np
from tqdm import tqdm
from dataset import CehrBertTokenizer, create_cehrbert_sequence

def materialize_split(parquet_path, vocab_path, output_dir, split_name, shard_size=1000):
    print(f"Materializing {split_name} split from {parquet_path}...")
    os.makedirs(output_dir, exist_ok=True)
    
    tokenizer = CehrBertTokenizer(vocab_path)
    
    # Identify subject column
    con = duckdb.connect()
    cols = con.execute(f"DESCRIBE SELECT * FROM '{parquet_path}' LIMIT 1").df()['column_name'].tolist()
    subject_col = 'subject_id' if 'subject_id' in cols else 'patient_id'
    
    subjects = con.execute(f"SELECT DISTINCT {subject_col} FROM '{parquet_path}'").df()[subject_col].tolist()
    print(f"Found {len(subjects)} patients in {split_name}.")
    
    shard_count = 0
    all_lengths = []
    total_tokens = 0
    total_clinical_tokens = 0
    patient_ids = []
    
    for i in tqdm(range(0, len(subjects), shard_size), desc=f"Writing {split_name} shards"):
        chunk_subs = subjects[i:i+shard_size]
        subs_str = ",".join([f"'{s}'" if isinstance(s, str) else str(s) for s in chunk_subs])
        df_chunk = con.execute(f"SELECT * FROM '{parquet_path}' WHERE {subject_col} IN ({subs_str})").df()
        grouped = df_chunk.groupby(subject_col)
        
        shard_data = []
        for subj in chunk_subs:
            if subj in grouped.groups:
                patient_df = grouped.get_group(subj)
                # Generate FULL sequence
                seq_dict = create_cehrbert_sequence(patient_df, tokenizer, max_seq_len=None, is_pretraining=False)
                
                # Calculate tokens
                input_ids = seq_dict["input_ids"]
                all_lengths.append(len(input_ids))
                total_tokens += len(input_ids)
                
                # Exclude all special tokens (VS, VE, ATT variants, CLS, PAD, UNK, MASK) for clinical token count
                special_ids = set()
                for tok in tokenizer.special_tokens:
                    special_ids.add(tokenizer.code_to_idx[tok])
                special_ids.add(tokenizer.pad_id)
                special_ids.add(tokenizer.unk_id)
                clinical = sum(1 for x in input_ids if x not in special_ids)
                total_clinical_tokens += clinical
                
                # Convert numpy arrays to Python lists so torch.save doesn't embed numpy globals
                shard_data.append({
                    "subject_id": subj,
                    "input_ids": input_ids.tolist() if hasattr(input_ids, 'tolist') else list(input_ids),
                    "segment_ids": seq_dict["segment_ids"].tolist() if hasattr(seq_dict["segment_ids"], 'tolist') else list(seq_dict["segment_ids"]),
                    "time_stamps": seq_dict["time_stamps"].tolist() if hasattr(seq_dict["time_stamps"], 'tolist') else list(seq_dict["time_stamps"]),
                    "ages": seq_dict["ages"].tolist() if hasattr(seq_dict["ages"], 'tolist') else list(seq_dict["ages"]),
                    "attention_mask": seq_dict["attention_mask"].tolist() if hasattr(seq_dict["attention_mask"], 'tolist') else list(seq_dict["attention_mask"])
                })
                patient_ids.append(subj)
                
        shard_path = os.path.join(output_dir, f"shard_{shard_count:05d}.pt")
        torch.save(shard_data, shard_path)
        shard_count += 1
        
    con.close()
    
    manifest = {
        "split": split_name,
        "number_of_patients": len(patient_ids),
        "number_of_shards": shard_count,
        "patients_per_shard": shard_size,
        "total_clinical_tokens": total_clinical_tokens,
        "total_encoded_tokens": total_tokens,
        "max_sequence_length": int(np.max(all_lengths)) if all_lengths else 0,
        "median_sequence_length": float(np.median(all_lengths)) if all_lengths else 0,
        "mean_sequence_length": float(np.mean(all_lengths)) if all_lengths else 0,
        "patient_ids": patient_ids
    }
    
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Materialized {split_name} -> {manifest_path}")

def main():
    base_dir = "artifacts/cehrbert_pretrain_sequences"
    train_path = "data/processed/train_events.parquet"
    val_path = "data/processed/val_events.parquet"
    vocab_path = "data/processed/code_vocab.json"
    
    materialize_split(train_path, vocab_path, os.path.join(base_dir, "train"), "train")
    materialize_split(val_path, vocab_path, os.path.join(base_dir, "val"), "val")

if __name__ == "__main__":
    main()
