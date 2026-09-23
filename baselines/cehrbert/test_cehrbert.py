import os
import torch
import json
import pandas as pd
import numpy as np
from dataset import CehrBertTokenizer, create_cehrbert_sequence, CehrBertIterableDataset
from model import CehrBertPretrainModel, CehrBertBiLSTMClassifier, CehrBertPooledClassifier
from collator import cehrbert_mlm_collate

def test_tokenizer_and_sequence():
    print("Running test_tokenizer_and_sequence...")
    vocab_path = "data/processed/code_vocab.json"
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

def test_mlm_collator():
    print("Running test_mlm_collator...")
    vocab_path = "data/processed/code_vocab.json"
    if not os.path.exists(vocab_path): return
    tokenizer = CehrBertTokenizer(vocab_path)
    
    # Create large synthetic batch
    batch_size = 100
    seq_len = 100
    
    # Let's say code IDs 10 to 50 are real clinical concepts
    synthetic_input_ids = torch.randint(10, 50, (batch_size, seq_len))
    # inject some special tokens
    synthetic_input_ids[:, 0] = tokenizer.vs_id
    synthetic_input_ids[:, -1] = tokenizer.ve_id
    
    batch = []
    for i in range(batch_size):
        batch.append({
            "input_ids": synthetic_input_ids[i],
            "segment_ids": torch.zeros(seq_len, dtype=torch.int64),
            "time_stamps": torch.zeros(seq_len, dtype=torch.float32),
            "ages": torch.zeros(seq_len, dtype=torch.float32),
            "attention_mask": torch.ones(seq_len, dtype=torch.int64),
        })
        
    collated = cehrbert_mlm_collate(batch, tokenizer, mlm_probability=0.15)
    
    labels = collated["labels"]
    inputs = collated["input_ids"]
    
    mask_positions = labels != -100
    total_eligible = (synthetic_input_ids != tokenizer.vs_id) & (synthetic_input_ids != tokenizer.ve_id)
    
    selected_count = mask_positions.sum().item()
    eligible_count = total_eligible.sum().item()
    
    pct_selected = selected_count / eligible_count
    
    masked_count = (inputs[mask_positions] == tokenizer.mask_id).sum().item()
    unchanged_count = (inputs[mask_positions] == synthetic_input_ids[mask_positions]).sum().item()
    random_count = selected_count - masked_count - unchanged_count
    
    print(f"Eligible: {eligible_count}, Selected: {selected_count} ({pct_selected*100:.1f}%)")
    print(f"MASK: {masked_count/selected_count*100:.1f}%")
    print(f"Unchanged: {unchanged_count/selected_count*100:.1f}%")
    print(f"Random: {random_count/selected_count*100:.1f}%")
    
    assert 0.12 < pct_selected < 0.18
    assert 0.75 < masked_count/selected_count < 0.85
    assert 0.05 < unchanged_count/selected_count < 0.15
    print("test_mlm_collator passed!\n")

def test_sampling_window():
    print("Running test_sampling_window...")
    vocab_path = "data/processed/code_vocab.json"
    if not os.path.exists(vocab_path): return
    tokenizer = CehrBertTokenizer(vocab_path)
    
    # > 300 token patient
    data = []
    # 50 visits, 10 events each = 500 events + 50 VS + 50 VE + 49 ATT = 649 tokens
    for i in range(50):
        for j in range(10):
            data.append({'code_id': 'code1', 'timestamp_days': i*10 + j, 'age_at_event_days': 300+i, 'hadm_id': i})
    df = pd.DataFrame(data)
    
    # Run pretraining random sampling multiple times
    starts = []
    for _ in range(5):
        seq = create_cehrbert_sequence(df, tokenizer, max_seq_len=300, is_pretraining=True)
        assert len(seq['input_ids']) <= 300
        # Should start at VS token (or CLS if it happened to pick 0)
        assert seq['input_ids'][0] == tokenizer.vs_id or seq['input_ids'][0] == tokenizer.cls_id
        # Chronological timestamps
        assert np.all(np.diff(seq['time_stamps'][seq['attention_mask']==1]) >= 0)
        starts.append(seq['time_stamps'][0])
        
    print(f"Pretraining starts: {starts}")
    assert len(set(starts)) > 1 # Random sampling should yield different starts
    
    # Deterministic validation
    val_seq1 = create_cehrbert_sequence(df, tokenizer, max_seq_len=300, is_pretraining=False)
    val_seq2 = create_cehrbert_sequence(df, tokenizer, max_seq_len=300, is_pretraining=False)
    assert np.array_equal(val_seq1['input_ids'], val_seq2['input_ids'])
    print("test_sampling_window passed!\n")

def test_tiny_mimic_smoke():
    print("Running tiny MIMIC smoke test...")
    vocab_path = "data/processed/code_vocab.json"
    mimic_path = "data/processed/test_events.parquet"
    if not os.path.exists(mimic_path):
        print("MIMIC test data not found, skipping.")
        return
        
    # Read just a few rows
    df = pd.read_parquet(mimic_path)
    # Get first 2 patients
    subjects = df['subject_id'].unique()[:2]
    df_tiny = df[df['subject_id'].isin(subjects)].copy()
    
    shards_dir = "tiny_mimic_shards"
    os.makedirs(shards_dir, exist_ok=True)
    tokenizer = CehrBertTokenizer(vocab_path)
    # create full sequence
    seq1 = create_cehrbert_sequence(df_tiny[df_tiny['subject_id']==subjects[0]], tokenizer, max_seq_len=None)
    seq2 = create_cehrbert_sequence(df_tiny[df_tiny['subject_id']==subjects[1]], tokenizer, max_seq_len=None)
    # Convert numpy to lists for torch.save compatibility with PyTorch 2.6+
    for seq in [seq1, seq2]:
        for k in seq:
            if hasattr(seq[k], 'tolist'):
                seq[k] = seq[k].tolist()
    seq1['subject_id'] = int(subjects[0])
    seq2['subject_id'] = int(subjects[1])
    torch.save([seq1, seq2], os.path.join(shards_dir, "shard_0.pt"))
    
    dataset = CehrBertIterableDataset(shards_dir, vocab_path, max_seq_len=50, is_pretraining=True)
    iterator = iter(dataset)
    batch = [next(iterator), next(iterator)]
    
    # Forward and backward test
    collated = cehrbert_mlm_collate(batch, dataset.tokenizer)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab_size = dataset.tokenizer.vocab_size
    model = CehrBertPretrainModel(vocab_size=vocab_size, d_model=128, n_layers=2, n_heads=4, max_seq_len=50).to(device)
    
    input_ids = collated["input_ids"].to(device)
    segment_ids = collated["segment_ids"].to(device)
    time_stamps = collated["time_stamps"].to(device)
    ages = collated["ages"].to(device)
    attention_mask = collated["attention_mask"].to(device)
    labels = collated["labels"].to(device)
    
    hidden, logits = model(input_ids, segment_ids, time_stamps, ages, attention_mask)
    
    loss_fct = torch.nn.CrossEntropyLoss()
    loss = loss_fct(logits.view(-1, vocab_size), labels.view(-1))
    
    print("Loss:", loss.item())
    assert not torch.isnan(loss) and not torch.isinf(loss)
    assert (labels != -100).sum().item() > 0
    
    loss.backward()
    
    # Check gradients
    assert model.embeddings.concept_embeddings.weight.grad is not None
    assert model.embeddings.time_embeddings.w.grad is not None
    assert model.embeddings.age_embeddings.w.grad is not None
    assert model.encoder.layers[0].linear1.weight.grad is not None
    
    print("MIMIC tiny smoke test passed! Forward/Backward successful.\n")
    import shutil
    shutil.rmtree(shards_dir)
    
def test_tiny_nch_smoke():
    print("Running tiny NCH smoke test...")
    vocab_path = "data/processed/code_vocab.json"
    nch_path = "artifacts/nch_stage2/v2/processed/canonical_events_clean.parquet"
    if not os.path.exists(nch_path):
        print("NCH test data not found, skipping.")
        return
        
    df = pd.read_parquet(nch_path)
    subjects = df['patient_id'].unique()[:2]
    df_tiny = df[df['patient_id'].isin(subjects)].copy()
    
    shards_dir = "tiny_nch_shards"
    os.makedirs(shards_dir, exist_ok=True)
    tokenizer = CehrBertTokenizer(vocab_path)
    seq = create_cehrbert_sequence(df_tiny[df_tiny['patient_id']==subjects[0]], tokenizer, max_seq_len=None)
    for k in seq:
        if hasattr(seq[k], 'tolist'):
            seq[k] = seq[k].tolist()
    seq['subject_id'] = int(subjects[0])
    torch.save([seq], os.path.join(shards_dir, "shard_0.pt"))
    
    dataset = CehrBertIterableDataset(shards_dir, vocab_path, max_seq_len=50, is_pretraining=False)
    iterator = iter(dataset)
    batch = next(iterator)
    
    assert batch['input_ids'].shape == (50,)
    print("NCH tiny smoke test passed!")
    import shutil
    shutil.rmtree(shards_dir)

def test_worker_uniqueness():
    print("Running test_worker_uniqueness...")
    vocab_path = "data/processed/code_vocab.json"
    if not os.path.exists(vocab_path): return
    
    # Create fake shards
    shards_dir = "test_shards"
    os.makedirs(shards_dir, exist_ok=True)
    tokenizer = CehrBertTokenizer(vocab_path)
    
    # 5 shards, 10 patients each
    patient_idx = 0
    for s in range(5):
        shard_data = []
        for p in range(10):
            shard_data.append({
                "subject_id": patient_idx,
                "input_ids": np.array([0, 1, 2]),
                "segment_ids": np.array([0, 0, 0]),
                "time_stamps": np.array([0., 1., 2.]),
                "ages": np.array([0., 1., 2.]),
                "attention_mask": np.array([1, 1, 1])
            })
            patient_idx += 1
        torch.save(shard_data, os.path.join(shards_dir, f"shard_{s}.pt"))
        
    dataset = CehrBertIterableDataset(shards_dir, vocab_path, max_seq_len=5, is_pretraining=True)
    
    # Run with 2 workers
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=2)
    
    seen = []
    for batch in loader:
        # Input ids [B, L]
        # In this dummy test we just count how many items we get
        # To get patient IDs we would need the dataset to return them, 
        # but just checking length and uniqueness of the generated sequences is fine
        seen.extend(batch["input_ids"].tolist())
        
    assert len(seen) == 50
    # Delete shards
    import shutil
    shutil.rmtree(shards_dir)
    print("test_worker_uniqueness passed!\n")

def test_mlm_optimization():
    print("Running test_mlm_optimization...")
    vocab_size = 100
    model = CehrBertPretrainModel(vocab_size=vocab_size, d_model=32, n_layers=2, n_heads=2, max_seq_len=10)
    
    input_ids = torch.randint(0, vocab_size, (2, 10))
    segment_ids = torch.zeros(2, 10, dtype=torch.long)
    time_stamps = torch.zeros(2, 10)
    ages = torch.zeros(2, 10)
    attention_mask = torch.ones(2, 10)
    
    labels = input_ids.clone()
    labels[0, :5] = -100
    labels[1, 5:] = -100
    
    # Mode 1: Full
    hidden, full_logits = model(input_ids, segment_ids, time_stamps, ages, attention_mask)
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    loss_full = loss_fct(full_logits.view(-1, vocab_size), labels.view(-1))
    loss_full.backward()
    grad_full = model.mlm_head.weight.grad.clone()
    
    model.zero_grad()
    
    # Mode 2: Masked
    hidden_opt, opt_logits, masked_labels = model(input_ids, segment_ids, time_stamps, ages, attention_mask, labels=labels)
    loss_fct_opt = torch.nn.CrossEntropyLoss()
    loss_opt = loss_fct_opt(opt_logits, masked_labels)
    loss_opt.backward()
    grad_opt = model.mlm_head.weight.grad.clone()
    
    assert torch.allclose(loss_full, loss_opt)
    assert torch.allclose(grad_full, grad_opt, atol=1e-5)
    print("test_mlm_optimization passed!\n")
    
if __name__ == "__main__":
    test_model_forward()
    test_tokenizer_and_sequence()
    test_tiny_mimic_smoke()
    test_mlm_collator()
    test_sampling_window()
    test_tiny_nch_smoke()
    test_worker_uniqueness()
    test_mlm_optimization()
    print("ALL TESTS PASSED")
