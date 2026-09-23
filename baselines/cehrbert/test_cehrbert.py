import os
import torch
import json
import pandas as pd
from dataset import CehrBertTokenizer, create_cehrbert_sequence, CehrBertDataset
from model import CehrBertPretrainModel, CehrBertBiLSTMClassifier, CehrBertPooledClassifier

def test_tokenizer_and_sequence():
    print("Running test_tokenizer_and_sequence...")
    vocab_path = "../../data/processed/code_vocab.json"
    if not os.path.exists(vocab_path):
        print("vocab not found, skipping.")
        return
        
    tokenizer = CehrBertTokenizer(vocab_path)
    
    # Create fake patient df
    data = [
        {'code_id': 'code1', 'timestamp_days': 0.0, 'age_at_event_days': 300, 'hadm_id': 1},
        {'code_id': 'code2', 'timestamp_days': 1.0, 'age_at_event_days': 301, 'hadm_id': 1},
        {'code_id': 'code3', 'timestamp_days': 20.0, 'age_at_event_days': 320, 'hadm_id': 2},
        {'code_id': 'code4', 'timestamp_days': 55.0, 'age_at_event_days': 355, 'hadm_id': 3},
        {'code_id': 'code5', 'timestamp_days': 450.0, 'age_at_event_days': 750, 'hadm_id': 4},
    ]
    df = pd.DataFrame(data)
    
    seq = create_cehrbert_sequence(df, tokenizer, max_seq_len=20)
    
    tokens = seq['input_ids'].tolist()
    decoded = [tokenizer.idx_to_code.get(t, '[PAD]') for t in tokens]
    print("Decoded sequence:", decoded)
    
    assert '[CLS]' in decoded
    assert '[VS]' in decoded
    assert '[VE]' in decoded
    assert 'W2' in decoded # 20-1=19 days -> W2
    assert 'M1' in decoded # 55-20=35 days -> M1
    assert 'LT' in decoded # 450-55=395 days -> LT
    
    print("test_tokenizer_and_sequence passed!")

def test_model_forward():
    print("Running test_model_forward...")
    vocab_size = 100
    model = CehrBertPretrainModel(vocab_size=vocab_size, d_model=16, n_layers=1, n_heads=2, max_seq_len=20)
    
    B, L = 2, 20
    input_ids = torch.randint(0, vocab_size, (B, L))
    segment_ids = torch.randint(0, 2, (B, L))
    time_stamps = torch.rand(B, L) * 100
    ages = torch.rand(B, L) * 100
    attention_mask = torch.ones(B, L, dtype=torch.int64)
    attention_mask[:, 10:] = 0
    
    hidden_states, mlm_logits = model(input_ids, segment_ids, time_stamps, ages, attention_mask)
    assert mlm_logits.shape == (B, L, vocab_size)
    
    bilstm_head = CehrBertBiLSTMClassifier(model, d_model=16, num_classes=5)
    logits = bilstm_head(input_ids, segment_ids, time_stamps, ages, attention_mask)
    assert logits.shape == (B, 5)

    mlp_head = CehrBertPooledClassifier(model, d_model=16, num_classes=5)
    logits_mlp = mlp_head(input_ids, segment_ids, time_stamps, ages, attention_mask)
    assert logits_mlp.shape == (B, 5)
    
    print("test_model_forward passed!")

def test_tiny_mimic_smoke():
    print("Running tiny MIMIC smoke test...")
    vocab_path = "../../data/processed/code_vocab.json"
    mimic_path = "../../data/processed/test_events.parquet"
    if not os.path.exists(mimic_path):
        print("MIMIC test data not found, skipping.")
        return
        
    # Read just a few rows
    df = pd.read_parquet(mimic_path)
    # Get first 2 patients
    subjects = df['subject_id'].unique()[:2]
    df_tiny = df[df['subject_id'].isin(subjects)].copy()
    
    tiny_path = "tiny_mimic.parquet"
    df_tiny.to_parquet(tiny_path)
    
    dataset = CehrBertDataset(tiny_path, vocab_path, max_seq_len=50)
    batch = dataset[0]
    
    assert batch['input_ids'].shape == (50,)
    print("MIMIC tiny smoke test passed!")
    
    os.remove(tiny_path)
    
def test_tiny_nch_smoke():
    print("Running tiny NCH smoke test...")
    vocab_path = "../../data/processed/code_vocab.json"
    nch_path = "../../artifacts/nch_stage2/v2/processed/canonical_events_clean.parquet"
    if not os.path.exists(nch_path):
        print("NCH test data not found, skipping.")
        return
        
    df = pd.read_parquet(nch_path)
    subjects = df['patient_id'].unique()[:2]
    df_tiny = df[df['patient_id'].isin(subjects)].copy()
    
    tiny_path = "tiny_nch.parquet"
    df_tiny.to_parquet(tiny_path)
    
    dataset = CehrBertDataset(tiny_path, vocab_path, max_seq_len=50)
    batch = dataset[0]
    
    assert batch['input_ids'].shape == (50,)
    print("NCH tiny smoke test passed!")
    os.remove(tiny_path)

if __name__ == "__main__":
    test_model_forward()
    test_tokenizer_and_sequence()
    test_tiny_mimic_smoke()
    test_tiny_nch_smoke()
    print("ALL TESTS PASSED")
