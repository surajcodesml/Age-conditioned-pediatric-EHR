import json
import os
import torch
import numpy as np
from torch.utils.data import Dataset
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
    max_seq_len: int = 300
) -> Dict[str, np.ndarray]:
    """
    Given a dataframe of events for a single patient, ordered by time,
    construct the CEHR-BERT representation.
    """
    # events_df should have columns: code_id, timestamp_days, age_at_event_days, hadm_id (optional)
    
    # Sort just in case
    events_df = events_df.sort_values('timestamp_days')
    
    tokens = [tokenizer.cls_id]
    segment_ids = [0]
    time_stamps = [events_df.iloc[0]['timestamp_days'] if len(events_df)>0 else 0.0]
    ages = [events_df.iloc[0]['age_at_event_days'] if len(events_df)>0 else 0.0]
    
    # Group by hadm_id for visit segmentation. If missing, treat same day as same visit.
    if 'hadm_id' in events_df.columns:
        events_df['visit_group'] = events_df['hadm_id'].fillna(-1).astype(str)
        # For missing hadm_id (-1), group by day
        missing_mask = events_df['visit_group'] == '-1.0'
        if missing_mask.any():
            events_df.loc[missing_mask, 'visit_group'] = 'day_' + (events_df.loc[missing_mask, 'timestamp_days'] // 1).astype(str)
    else:
        # If no hadm_id, group by day
        events_df['visit_group'] = 'day_' + (events_df['timestamp_days'] // 1).astype(str)
        
    visit_groups = events_df.groupby('visit_group', sort=False)
    
    current_segment = 0
    last_visit_end_time = None
    
    for visit_id, group in visit_groups:
        visit_start_time = group['timestamp_days'].min()
        visit_age = group['age_at_event_days'].min()
        
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
            tokens.append(tokenizer.encode(row['code_id']))
            segment_ids.append(current_segment)
            time_stamps.append(row['timestamp_days'])
            ages.append(row['age_at_event_days'])
            
        # [VE] token
        visit_end_time = group['timestamp_days'].max()
        visit_end_age = group['age_at_event_days'].max()
        tokens.append(tokenizer.ve_id)
        segment_ids.append(current_segment)
        time_stamps.append(visit_end_time)
        ages.append(visit_end_age)
        
        last_visit_end_time = visit_end_time
        
        if len(tokens) >= max_seq_len:
            break
            
    # Truncate
    if len(tokens) > max_seq_len:
        # CEHR-BERT paper randomly crops subsequences for long patients, or tail truncate
        # For simplicity in this function, we will tail truncate, but training script can implement random crop
        tokens = tokens[-max_seq_len:]
        segment_ids = segment_ids[-max_seq_len:]
        time_stamps = time_stamps[-max_seq_len:]
        ages = ages[-max_seq_len:]
        
    # Pad
    pad_len = max_seq_len - len(tokens)
    attention_mask = [1] * len(tokens) + [0] * pad_len
    
    tokens = tokens + [tokenizer.pad_id] * pad_len
    segment_ids = segment_ids + [0] * pad_len
    time_stamps = time_stamps + [0.0] * pad_len
    ages = ages + [0.0] * pad_len
    
    return {
        "input_ids": np.array(tokens, dtype=np.int64),
        "segment_ids": np.array(segment_ids, dtype=np.int64),
        "time_stamps": np.array(time_stamps, dtype=np.float32),
        "ages": np.array(ages, dtype=np.float32),
        "attention_mask": np.array(attention_mask, dtype=np.int64)
    }

class CehrBertDataset(Dataset):
    def __init__(self, parquet_path, vocab_path, max_seq_len=300):
        self.df = pd.read_parquet(parquet_path)
        self.tokenizer = CehrBertTokenizer(vocab_path)
        self.max_seq_len = max_seq_len
        
        if 'subject_id' in self.df.columns:
            self.subject_col = 'subject_id'
        else:
            self.subject_col = 'patient_id'
            
        self.subject_ids = self.df[self.subject_col].unique()
        # Create an index mapping for fast groupby retrieval if memory allows
        # Or just use grouped objects
        self.grouped = self.df.groupby(self.subject_col)

    def __len__(self):
        return len(self.subject_ids)

    def __getitem__(self, idx):
        subject_id = self.subject_ids[idx]
        patient_df = self.grouped.get_group(subject_id)
        
        seq_dict = create_cehrbert_sequence(patient_df, self.tokenizer, self.max_seq_len)
        
        return {
            "input_ids": torch.tensor(seq_dict["input_ids"]),
            "segment_ids": torch.tensor(seq_dict["segment_ids"]),
            "time_stamps": torch.tensor(seq_dict["time_stamps"]),
            "ages": torch.tensor(seq_dict["ages"]),
            "attention_mask": torch.tensor(seq_dict["attention_mask"])
        }
