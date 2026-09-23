import os
import torch
import time
import psutil
from model import CehrBertPretrainModel
from dataset import CehrBertTokenizer

def run_preflight():
    print("CEHR-BERT Preflight Hardware Benchmark")
    print("="*40)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA not available. Exiting preflight.")
        return
        
    print(f"Device: {torch.cuda.get_device_name(device)}")
    total_vram = torch.cuda.get_device_properties(device).total_memory / (1024**2)
    print(f"Total VRAM: {total_vram:.0f} MB")
    
    vocab_size = 30635
    model = CehrBertPretrainModel(
        vocab_size=vocab_size,
        d_model=128,
        n_layers=5,
        n_heads=8,
        max_seq_len=300
    ).to(device)
    
    loss_fct = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    
    # Try batch sizes
    candidates = [32, 16, 8]
    best_config = None
    
    for bs in candidates:
        print(f"\nTesting Microbatch Size: {bs}")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Synthetic worst-case batch (all tokens real)
        input_ids = torch.randint(0, vocab_size, (bs, 300)).to(device)
        segment_ids = torch.zeros(bs, 300, dtype=torch.long).to(device)
        time_stamps = torch.zeros(bs, 300, dtype=torch.float).to(device)
        ages = torch.zeros(bs, 300, dtype=torch.float).to(device)
        attention_mask = torch.ones(bs, 300, dtype=torch.long).to(device)
        labels = input_ids.clone()
        labels[:, :255] = -100 # 15% masked
        
        try:
            # warmup
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                h, logits, masked_labels = model(input_ids, segment_ids, time_stamps, ages, attention_mask, labels=labels)
                loss = loss_fct(logits, masked_labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            torch.cuda.synchronize()
            t0 = time.time()
            
            # benchmark 10 steps
            for _ in range(10):
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    h, logits, masked_labels = model(input_ids, segment_ids, time_stamps, ages, attention_mask, labels=labels)
                    loss = loss_fct(logits, masked_labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
            torch.cuda.synchronize()
            elapsed = time.time() - t0
            steps_per_sec = 10 / elapsed
            tokens_per_sec = (bs * 300 * 10) / elapsed
            
            peak_vram = torch.cuda.max_memory_allocated(device) / (1024**2)
            pct_vram = (peak_vram / total_vram) * 100
            
            print(f"  Peak VRAM: {peak_vram:.0f} MB ({pct_vram:.1f}%)")
            print(f"  Steps/sec: {steps_per_sec:.1f}")
            print(f"  Tokens/sec: {tokens_per_sec:.0f}")
            
            if pct_vram < 90.0:
                print(f"  -> SUCCESS (Fits safely)")
                if best_config is None:
                    best_config = bs
            else:
                print(f"  -> WARNING (Too close to VRAM limit)")
                
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  -> OOM FAILED")
                torch.cuda.empty_cache()
            else:
                raise e
                
    if best_config is not None:
        grad_accum = 32 // best_config
        print(f"\nRecommended production configuration:")
        print(f"--micro_batch_size {best_config} --gradient_accumulation_steps {grad_accum}")
    else:
        print("\nAll tested microbatch sizes failed to fit safely.")

if __name__ == "__main__":
    run_preflight()
