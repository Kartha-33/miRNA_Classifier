"""
Training script for Hybrid miRNA Classification Model
"""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
from tqdm import tqdm

from .config import ModelConfig
from .dataset import MirNADataset, collate_hybrid_batch
from .model import HybridMirNA


def compute_metrics(predictions, labels):
    """Compute classification metrics."""
    preds = predictions.argmax(dim=1).cpu().numpy()
    labels = labels.cpu().numpy()
    
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )
    
    # ROC-AUC (using probabilities)
    try:
        probs = torch.softmax(predictions, dim=1)[:, 1].cpu().numpy()
        auc = roc_auc_score(labels, probs)
    except:
        auc = 0.0
    
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    criterion = nn.CrossEntropyLoss()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        graph_x = batch['graph_x'].to(device)
        graph_edge_index = batch['graph_edge_index'].to(device)
        graph_batch = batch['graph_batch'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            graph_x=graph_x,
            graph_edge_index=graph_edge_index,
            graph_batch=graph_batch
        )
        
        # Compute loss
        loss = criterion(logits, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Track metrics
        total_loss += loss.item()
        all_preds.append(logits.detach())
        all_labels.append(labels)
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    # Compute epoch metrics
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_preds, all_labels)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics


def evaluate(model, dataloader, device, split_name="Val"):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"[{split_name}]")
        for batch in pbar:
            # Move to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            graph_x = batch['graph_x'].to(device)
            graph_edge_index = batch['graph_edge_index'].to(device)
            graph_batch = batch['graph_batch'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                graph_x=graph_x,
                graph_edge_index=graph_edge_index,
                graph_batch=graph_batch
            )
            
            # Compute loss
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            all_preds.append(logits)
            all_labels.append(labels)
    
    # Compute metrics
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    metrics = compute_metrics(all_preds, all_labels)
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics


def main(args):
    # Configuration
    config = ModelConfig()
    config.batch_size = args.batch_size
    config.num_epochs = args.num_epochs
    config.device = args.device
    
    # Set device with MPS support for Apple Silicon
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("🚀 Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    full_dataset = MirNADataset(
        csv_path=args.data_path,
        config=config,
        predict_structure=args.predict_structure
    )
    
    # Split into train/val/test
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_hybrid_batch,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_hybrid_batch,
        num_workers=args.num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_hybrid_batch,
        num_workers=args.num_workers
    )
    
    # Initialize model
    print("\nInitializing model...")
    model = HybridMirNA(config).to(device)
    
    # Optimizer with different learning rates
    param_groups = model.get_separate_parameter_groups()
    optimizer = AdamW(param_groups, weight_decay=config.weight_decay)
    
    # Learning rate scheduler
    total_steps = len(train_loader) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    print("\n" + "="*50)
    print("Starting Training")
    print("="*50)
    
    best_val_f1 = 0.0
    history = {'train': [], 'val': []}
    
    for epoch in range(1, config.num_epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, device, epoch)
        
        # Validate
        val_metrics = evaluate(model, val_loader, device, split_name="Val")
        
        # Log
        print(f"\nEpoch {epoch}/{config.num_epochs}")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Acc: {val_metrics['accuracy']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}, "
              f"AUC: {val_metrics['auc']:.4f}")
        
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)
        
        # Save best model
        if val_metrics['f1'] > best_val_f1 or epoch == 1:  # Always save first epoch
            best_val_f1 = max(val_metrics['f1'], best_val_f1)
            os.makedirs(args.output_dir, exist_ok=True)
            checkpoint_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'config': config.__dict__
            }, checkpoint_path)
            print(f"  ✓ Saved best model (F1: {best_val_f1:.4f})")
    
    # Final evaluation on test set
    print("\n" + "="*50)
    print("Final Evaluation on Test Set")
    print("="*50)
    
    # Load best model
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, device, split_name="Test")
    
    print(f"\nTest Results:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1 Score:  {test_metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {test_metrics['auc']:.4f}")
    
    # Save results
    results = {
        'test_metrics': test_metrics,
        'best_val_f1': best_val_f1,
        'history': history
    }
    
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        # Convert numpy types to native Python types
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(item) for item in obj]
            return obj
        
        json.dump(convert(results), f, indent=2)
    
    print(f"\n✓ Results saved to {results_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train HybridMirNA model')
    
    # Data
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to CSV file with sequence/structure/label')
    parser.add_argument('--predict_structure', action='store_true',
                        help='Use RNAfold to predict missing structures')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='./outputs')
    
    args = parser.parse_args()
    main(args)
