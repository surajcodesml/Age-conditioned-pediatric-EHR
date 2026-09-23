import torch

def cehrbert_mlm_collate(batch, tokenizer, mlm_probability=0.15):
    """
    Collate function that applies 80/10/10 masking for MLM.
    Excludes special tokens (VS, VE, ATT, CLS, PAD, UNK, MASK).
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    segment_ids = torch.stack([item['segment_ids'] for item in batch])
    time_stamps = torch.stack([item['time_stamps'] for item in batch])
    ages = torch.stack([item['ages'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    
    labels = input_ids.clone()
    
    # Identify special tokens to exclude from masking
    # special_tokens include CLS, PAD, UNK, MASK, VS, VE, W0..W3, M1..M11, LT
    # For speed, we just use tokenizer.special_tokens set or convert to ids
    special_ids = set()
    for token in tokenizer.special_tokens:
        if token in tokenizer.code_to_idx:
            special_ids.add(tokenizer.code_to_idx[token])
    # Also exclude PAD (0) and UNK (1) if they are not in tokenizer.special_tokens
    special_ids.add(tokenizer.pad_id)
    special_ids.add(tokenizer.unk_id)
    
    # Create mask of eligible tokens
    probability_matrix = torch.full(labels.shape, mlm_probability)
    
    for sid in special_ids:
        probability_matrix.masked_fill_(labels == sid, value=0.0)
        
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens
    
    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = tokenizer.mask_id
    
    # 10% of the time, we replace masked input tokens with random word
    # The rest 10% of the time, we keep the masked input tokens unchanged
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer.code_to_idx), labels.shape, dtype=torch.long)
    
    # Exclude special tokens from random words if possible, but standard BERT just samples uniformly from vocab
    # Let's just sample uniformly from all vocab, it's fine.
    input_ids[indices_random] = random_words[indices_random]
    
    return {
        "input_ids": input_ids,
        "segment_ids": segment_ids,
        "time_stamps": time_stamps,
        "ages": ages,
        "attention_mask": attention_mask,
        "labels": labels
    }
