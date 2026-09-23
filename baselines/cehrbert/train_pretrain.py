import argparse
import os
import torch
import psutil
import time
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CehrBertIterableDataset, CehrBertTokenizer
from collator import cehrbert_mlm_collate
from model import CehrBertPretrainModel

def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain CEHR-BERT Stage-1")
    parser.add_argument("--model", type=str, default="cehrbert_pang2021")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=5)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=300)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--micro_batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--data_dir", type=str, default="artifacts/cehrbert_pretrain_sequences")
    parser.add_argument("--output_dir", type=str, default="artifacts/cehrbert_pretrain")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()

def print_diagnostics(args, device, train_dataset, val_dataset):
    print("=" * 50)
    print("STARTUP DIAGNOSTICS")
    print(f"device: {device}")
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(device)}")
    print(f"precision: {args.amp}")
    print(f"physical microbatch: {args.micro_batch_size}")
    print(f"gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"effective batch: {args.micro_batch_size * args.gradient_accumulation_steps}")
    print(f"num workers: {args.num_workers}")
    
    # We could count patients from manifest
    import json
    train_manifest = os.path.join(args.data_dir, "train", "manifest.json")
    val_manifest = os.path.join(args.data_dir, "val", "manifest.json")
    train_patients = 0
    if os.path.exists(train_manifest):
        with open(train_manifest, 'r') as f:
            train_patients = json.load(f)["number_of_patients"]
    val_patients = 0
    if os.path.exists(val_manifest):
        with open(val_manifest, 'r') as f:
            val_patients = json.load(f)["number_of_patients"]
            
    print(f"number train patients: {train_patients}")
    print(f"number val patients: {val_patients}")
    
    effective_bs = args.micro_batch_size * args.gradient_accumulation_steps
    steps_per_epoch = (train_patients + effective_bs - 1) // effective_bs
    print(f"optimizer steps / epoch: {steps_per_epoch}")
    print(f"total optimizer steps: {steps_per_epoch * args.epochs}")
    print("=" * 50)

def main():
    args = parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    vocab_path = "data/processed/code_vocab.json"
    train_dir = os.path.join(args.data_dir, "train")
    val_dir = os.path.join(args.data_dir, "val")
    
    # Use CehrBertIterableDataset, but we will instantiate it per epoch to pass the epoch seed
    tokenizer = CehrBertTokenizer(vocab_path)
    
    model = CehrBertPretrainModel(
        vocab_size=tokenizer.vocab_size,
        d_model=args.hidden_size,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        max_seq_len=args.max_seq_len
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of parameters: {num_params}")
    
    print_diagnostics(args, device, None, None)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Manifest counts
    import json
    train_manifest = os.path.join(train_dir, "manifest.json")
    train_patients = 154418
    if os.path.exists(train_manifest):
        with open(train_manifest, 'r') as f:
            train_patients = json.load(f)["number_of_patients"]
            
    effective_bs = args.micro_batch_size * args.gradient_accumulation_steps
    steps_per_epoch = max(1, train_patients // effective_bs)
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    
    scaler = None
    if args.amp == "fp16":
        scaler = torch.cuda.amp.GradScaler()
    elif args.amp == "bf16":
        # GradScaler not strictly needed for bf16, but can be used. We'll omit it for bf16 if not needed, 
        # or just use it. Let's use it if available.
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        
    start_epoch = 0
    global_step = 0
    
    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
        global_step = checkpoint.get('global_step', start_epoch * steps_per_epoch)
        
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    autocast_dtype = dtype_map[args.amp]
    
    loss_fct = torch.nn.CrossEntropyLoss()
    
    process = psutil.Process(os.getpid())
    
    for epoch in range(start_epoch, args.epochs):
        train_dataset = CehrBertIterableDataset(train_dir, vocab_path, max_seq_len=args.max_seq_len, is_pretraining=True, epoch=epoch)
        collate_fn = lambda batch: cehrbert_mlm_collate(batch, tokenizer, mlm_probability=args.mlm_probability)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.micro_batch_size, 
            shuffle=False, 
            collate_fn=collate_fn, 
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=2 if args.num_workers > 0 else None
        )
        
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        t0 = time.time()
        data_t0 = time.time()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        batches_processed = 0
        
        for step, batch in enumerate(pbar):
            data_time = time.time() - data_t0
            gpu_t0 = time.time()
            
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            segment_ids = batch["segment_ids"].to(device, non_blocking=True)
            time_stamps = batch["time_stamps"].to(device, non_blocking=True)
            ages = batch["ages"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            
            with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=autocast_dtype, enabled=args.amp != "fp32"):
                hidden, opt_logits, masked_labels = model(input_ids, segment_ids, time_stamps, ages, attention_mask, labels=labels)
                loss = loss_fct(opt_logits, masked_labels)
                loss = loss / args.gradient_accumulation_steps
                
            if scaler is not None and args.amp == "fp16":
                scaler.scale(loss).backward()
            else:
                loss.backward()
                
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if scaler is not None and args.amp == "fp16":
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
            total_loss += loss.item() * args.gradient_accumulation_steps
            batches_processed += 1
            
            gpu_time = time.time() - gpu_t0
            tokens_processed = input_ids.numel()
            elapsed = time.time() - t0
            
            # Masked accuracy
            with torch.no_grad():
                preds = opt_logits.argmax(dim=-1)
                acc = (preds == masked_labels).float().mean().item()
            
            if step % 50 == 0:
                vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2) if torch.cuda.is_available() else 0
                rss_mb = process.memory_info().rss / (1024**2)
                pbar.set_postfix({
                    "loss": f"{loss.item()*args.gradient_accumulation_steps:.4f}",
                    "acc": f"{acc:.3f}",
                    "t/s": f"{tokens_processed/elapsed:.0f}",
                    "data(s)": f"{data_time:.3f}",
                    "gpu(s)": f"{gpu_time:.3f}",
                    "vram": f"{vram_mb:.0f}M",
                    "rss": f"{rss_mb:.0f}M"
                })
                
            t0 = time.time()
            data_t0 = time.time()
            
        avg_train_loss = total_loss / max(1, batches_processed)
        
        # Validation
        val_dataset = CehrBertIterableDataset(val_dir, vocab_path, max_seq_len=args.max_seq_len, is_pretraining=False)
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.micro_batch_size, 
            shuffle=False, 
            collate_fn=collate_fn, 
            num_workers=args.num_workers,
            pin_memory=True,
            prefetch_factor=2 if args.num_workers > 0 else None
        )
        
        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for batch in pbar_val:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                segment_ids = batch["segment_ids"].to(device, non_blocking=True)
                time_stamps = batch["time_stamps"].to(device, non_blocking=True)
                ages = batch["ages"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                
                with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu', dtype=autocast_dtype, enabled=args.amp != "fp32"):
                    hidden, opt_logits, masked_labels = model(input_ids, segment_ids, time_stamps, ages, attention_mask, labels=labels)
                    loss = loss_fct(opt_logits, masked_labels)
                
                val_loss += loss.item()
                val_batches += 1
                
        avg_val_loss = val_loss / max(1, val_batches)
        print(f"Epoch {epoch+1} Summary: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
        ckpt_dict = {
            'epoch': epoch + 1,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
        }
        if scaler is not None and args.amp == "fp16":
            ckpt_dict['scaler_state_dict'] = scaler.state_dict()
            
        torch.save(ckpt_dict, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    # Save final model
    final_path = os.path.join(args.output_dir, "cehrbert_mlm_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to {final_path}")
    print("Pretraining Complete!")

if __name__ == "__main__":
    main()
