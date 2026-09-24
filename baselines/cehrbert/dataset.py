import json
import os
import torch
import numpy as np
from torch.utils.data import Dataset, IterableDataset
import math
import glob
import json
import pandas as pd
from typing import List, Dict, Optional

# Special Tokens
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
CLS_TOKEN = "[CLS]"
MASK_TOKEN = "[MASK]"
VS_TOKEN = "[VS]"
VE_TOKEN = "[VE]"
LT_TOKEN = "LT"

# ATTs
WEEK_TOKENS = [f"W{i}" for i in range(4)] # W0, W1, W2, W3
MONTH_TOKENS = [f"M{i}" for i in range(1, 12)] # M1 ... M11

class CehrBertTokenizer:
    def __init__(self, vocab_path: str):
        with open(vocab_path, 'r') as f:
            self.code_to_idx = json.load(f)
            
        # Existing pipeline assumes PAD=0, UNK=1, real codes start at 2
        # Let's verify and append CEHR-BERT special tokens
        self.special_tokens = [CLS_TOKEN, MASK_TOKEN, VS_TOKEN, VE_TOKEN, LT_TOKEN] + WEEK_TOKENS + MONTH_TOKENS
        
        # Max index in existing vocab
        max_idx = max(self.code_to_idx.values())
        
        for token in self.special_tokens:
            if token not in self.code_to_idx:
                max_idx += 1
                self.code_to_idx[token] = max_idx
                
        self.idx_to_code = {v: k for k, v in self.code_to_idx.items()}
        self.vocab_size = max(self.code_to_idx.values()) + 1
        
        self.pad_id = 0
        self.unk_id = 1
        self.cls_id = self.code_to_idx[CLS_TOKEN]
        self.mask_id = self.code_to_idx[MASK_TOKEN]
        self.vs_id = self.code_to_idx[VS_TOKEN]
        self.ve_id = self.code_to_idx[VE_TOKEN]

    def encode(self, code: str) -> int:
        return self.code_to_idx.get(code, self.unk_id)

    def get_att_token(self, delta_days: float) -> str:
        if delta_days < 0:
            delta_days = 0 # Should not happen if sorted, but safety
        if delta_days < 28:
            w_idx = int(delta_days // 7)
            w_idx = min(w_idx, 3) # Cap at W3
            return f"W{w_idx}"
        elif delta_days <= 365:
            m_idx = int(delta_days // 30)
            # Ensure it fits in M1..M11.
            m_idx = max(1, min(m_idx, 11))
            return f"M{m_idx}"
        else:
            return LT_TOKEN

def create_cehrbert_sequence(
    events_df: pd.DataFrame, 
    tokenizer: CehrBertTokenizer, 
    max_seq_len: int = 300,
    is_pretraining: bool = False
) -> Dict[str, np.ndarray]:
    """
    Given a dataframe of events for a single patient, ordered by time,
    construct the CEHR-BERT representation.
    """
    # events_df should have columns: code_id, timestamp_days, age_at_event_days, hadm_id (optional)
    
    # Identify time column
    time_col = 'timestamp_days'
    if time_col not in events_df.columns:
        if 'age_at_event_days' in events_df.columns:
            time_col = 'age_at_event_days'
        elif 'age_in_days' in events_df.columns:
            time_col = 'age_in_days'
        else:
            # Fallback to any column with 'time' or 'age'
            candidates = [c for c in events_df.columns if 'time' in c or 'age' in c]
            if candidates:
                time_col = candidates[0]
                
    # Identify age column
    age_col = 'age_at_event_days'
    if age_col not in events_df.columns:
        if 'age_in_days' in events_df.columns:
            age_col = 'age_in_days'
        else:
            age_col = time_col
            
    events_df = events_df.sort_values(time_col)
    
    tokens = [tokenizer.cls_id]
    segment_ids = [0]
    time_stamps = [events_df.iloc[0][time_col] if len(events_df)>0 else 0.0]
    ages = [events_df.iloc[0][age_col] if len(events_df)>0 else 0.0]
    
    # Group by hadm_id for visit segmentation. If missing, treat same day as same visit.
    if 'hadm_id' in events_df.columns:
        # Step 1: Recover missing hadm_ids using admission intervals
        has_hadm = events_df['hadm_id'].notna()
        if has_hadm.any():
            admissions = events_df[has_hadm].groupby('hadm_id')[time_col].agg(['min', 'max']).to_dict(orient='index')
            grace_period = 1.0 # 1 day
            for idx, row in events_df[~has_hadm].iterrows():
                t = row[time_col]
                for hadm, bounds in admissions.items():
                    if bounds['min'] - grace_period <= t <= bounds['max'] + grace_period:
                        events_df.at[idx, 'hadm_id'] = hadm
                        break
                        
        events_df['visit_group'] = events_df['hadm_id'].fillna(-1).astype(str)
        # For missing hadm_id (-1), group by day
        missing_mask = events_df['visit_group'] == '-1.0'
        if missing_mask.any():
            events_df.loc[missing_mask, 'visit_group'] = 'day_' + (events_df.loc[missing_mask, time_col] // 1).astype(str)
    else:
        # If no hadm_id, group by day
        events_df['visit_group'] = 'day_' + (events_df[time_col] // 1).astype(str)
        
    visit_groups = events_df.groupby('visit_group', sort=False)
    
    current_segment = 0
    last_visit_end_time = None
    
    for visit_id, group in visit_groups:
        visit_start_time = group[time_col].min()
        visit_age = group[age_col].min()
        
        # Insert ATT if not the first visit
        if last_visit_end_time is not None:
            delta_days = visit_start_time - last_visit_end_time
            att_token = tokenizer.get_att_token(delta_days)
            tokens.append(tokenizer.encode(att_token))
            segment_ids.append(current_segment)
            time_stamps.append(visit_start_time)
            ages.append(visit_age)
            
            # Flip segment A/B
            current_segment = 1 - current_segment
            
        # [VS] token
        tokens.append(tokenizer.vs_id)
        segment_ids.append(current_segment)
        time_stamps.append(visit_start_time)
        ages.append(visit_age)
        
        # Clinical events
        for _, row in group.iterrows():
            # For NCH, the column might be code instead of code_id
            code = row['code_id'] if 'code_id' in row else row.get('code', None)
            if code is None:
                # find a column named code
                candidates = [c for c in row.index if 'code' in c]
                code = row[candidates[0]] if candidates else str(row.iloc[0])
                
            tokens.append(tokenizer.encode(code))
            segment_ids.append(current_segment)
            time_stamps.append(row[time_col])
            ages.append(row[age_col])
            
        # [VE] token
        visit_end_time = group[time_col].max()
        visit_end_age = group[age_col].max()
        tokens.append(tokenizer.ve_id)
        segment_ids.append(current_segment)
        time_stamps.append(visit_end_time)
        ages.append(visit_end_age)
        
        last_visit_end_time = visit_end_time
        
        if max_seq_len is not None and len(tokens) >= max_seq_len and not is_pretraining:
            # We don't break early for pretraining because we need the full sequence to sample from
            pass
            
    # Truncate and Sample
    if max_seq_len is not None and len(tokens) > max_seq_len:
        vs_indices = [i for i, t in enumerate(tokens) if t == tokenizer.vs_id]
        if is_pretraining:
            import random
            if vs_indices:
                # Pick a random VS token as the start
                start_idx = random.choice(vs_indices)
                # If the chosen start_idx leaves fewer than max_seq_len tokens, we could just pad,
                # but to maximize token usage, we can shift start_idx back if possible, 
                # or just use it as is (CEHR-BERT usually just takes [start_idx : start_idx + max_seq_len]).
                # Let's just use it as is to strictly obey "beginning at a valid visit boundary".
            else:
                start_idx = random.randint(0, len(tokens) - max_seq_len)
        else:
            # Deterministic for validation/finetuning: most recent history
            start_idx = len(tokens) - max_seq_len
            valid_vs = [i for i in vs_indices if i >= start_idx]
            if valid_vs:
                start_idx = valid_vs[0]
                
        end_idx = start_idx + max_seq_len
        tokens = tokens[start_idx:end_idx]
        segment_ids = segment_ids[start_idx:end_idx]
        time_stamps = time_stamps[start_idx:end_idx]
        ages = ages[start_idx:end_idx]
        
    # Pad
    if max_seq_len is not None:
        pad_len = max_seq_len - len(tokens)
        attention_mask = [1] * len(tokens) + [0] * pad_len
        
        tokens = tokens + [tokenizer.pad_id] * pad_len
        segment_ids = segment_ids + [0] * pad_len
        time_stamps = time_stamps + [0.0] * pad_len
        ages = ages + [0.0] * pad_len
    else:
        attention_mask = [1] * len(tokens)
    

    return {
        "input_ids": np.array(tokens, dtype=np.int64),
        "segment_ids": np.array(segment_ids, dtype=np.int64),
        "time_stamps": np.array(time_stamps, dtype=np.float32),
        "ages": np.array(ages, dtype=np.float32),
        "attention_mask": np.array(attention_mask, dtype=np.int64)
    }

def _to_list(x):
    """Convert numpy arrays / tensors to plain lists; pass through if already a list."""
    if isinstance(x, list):
        return x
    return x.tolist()

def pad_and_crop(seq_dict, max_seq_len, tokenizer, is_pretraining, seed=None):
    tokens = _to_list(seq_dict["input_ids"])
    segment_ids = _to_list(seq_dict["segment_ids"])
    time_stamps = _to_list(seq_dict["time_stamps"])
    ages = _to_list(seq_dict["ages"])
    
    if len(tokens) > max_seq_len:
        vs_indices = [i for i, t in enumerate(tokens) if t == tokenizer.vs_id]
        if is_pretraining:
            import random
            rng = random.Random(seed)
            if vs_indices:
                start_idx = rng.choice(vs_indices)
            else:
                start_idx = rng.randint(0, len(tokens) - max_seq_len)
        else:
            start_idx = len(tokens) - max_seq_len
            valid_vs = [i for i in vs_indices if i >= start_idx]
            if valid_vs:
                start_idx = valid_vs[0]
                
        end_idx = start_idx + max_seq_len
        tokens = tokens[start_idx:end_idx]
        segment_ids = segment_ids[start_idx:end_idx]
        time_stamps = time_stamps[start_idx:end_idx]
        ages = ages[start_idx:end_idx]
        
    pad_len = max_seq_len - len(tokens)
    attention_mask = [1] * len(tokens) + [0] * pad_len
    
    tokens = tokens + [tokenizer.pad_id] * pad_len
    segment_ids = segment_ids + [0] * pad_len
    time_stamps = time_stamps + [0.0] * pad_len
    ages = ages + [0.0] * pad_len
    
    return {
        "input_ids": torch.tensor(tokens, dtype=torch.long),
        "segment_ids": torch.tensor(segment_ids, dtype=torch.long),
        "time_stamps": torch.tensor(time_stamps, dtype=torch.float),
        "ages": torch.tensor(ages, dtype=torch.float),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long)
    }

class CehrBertIterableDataset(IterableDataset):
    def __init__(self, shards_dir, vocab_path, max_seq_len=300, is_pretraining=False, seed=42, epoch=0):
        self.shards_dir = shards_dir
        self.tokenizer = CehrBertTokenizer(vocab_path)
        self.max_seq_len = max_seq_len
        self.is_pretraining = is_pretraining
        self.seed = seed
        self.epoch = epoch
        
        self.shards = sorted(glob.glob(os.path.join(shards_dir, "shard_*.pt")))
        
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        import random
        
        # Seed logic per epoch and worker
        worker_id = worker_info.id if worker_info is not None else 0
        rng = random.Random(self.seed + self.epoch + worker_id)
        
        # Partition shards
        shards = list(self.shards)
        if self.is_pretraining:
            rng.shuffle(shards)
            
        if worker_info is not None:
            num_workers = worker_info.num_workers
            shards = [s for i, s in enumerate(shards) if i % num_workers == worker_id]
            
        for shard_path in shards:
            shard_data = torch.load(shard_path, map_location='cpu', weights_only=False)
            if self.is_pretraining:
                rng.shuffle(shard_data)
                
            for seq_dict in shard_data:
                # Seed for random crop so it varies by epoch and patient
                patient_seed = rng.randint(0, 2**32 - 1)
                yield pad_and_crop(seq_dict, self.max_seq_len, self.tokenizer, self.is_pretraining, patient_seed)
