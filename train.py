import math
import random
import string
import argparse
import wandb
import datetime
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm, trange
from model import BrainrotX
from dataset import BrainrotXDataset

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train a transformer model for text rotation')
    
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--dim_feedforward', type=int, default=1024, help='Feedforward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    parser.add_argument('--batch_size', type=int, default=8, help='Training batch size')
    parser.add_argument('--iterations', type=int, default=10, help='Number of training iterations')
    parser.add_argument('--seq_length', type=int, default=1024, help='Sequence length')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of training samples')
    parser.add_argument('--rot', type=int, default=26, help='Text rotation amount')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    
    parser.add_argument('--wandb_project', type=str, default='brainrot26', help='WandB project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='WandB entity name')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    
    args = parser.parse_args()
    
    args.run_name = f"{args.wandb_project}_d{args.d_model}_h{args.nhead}_l{args.num_layers}_r{args.rot}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return args

def save_model(model, args, iteration=None):
    """Save model checkpoint.
    
    Args:
        model: The transformer model
        args: Parsed command line arguments
        iteration: Current iteration number (optional)
    """
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': {
            'd_model': args.d_model,
            'nhead': args.nhead,
            'num_layers': args.num_layers,
            'dim_feedforward': args.dim_feedforward,
            'dropout': args.dropout,
            'rot': args.rot
        }
    }
    
    save_path = os.path.join(args.checkpoint_dir, f"{args.run_name}.pt")
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")
    
    if wandb.run is not None:
        artifact = wandb.Artifact(
            name=f'model-{args.run_name}',
            type='model',
            description=f'Model checkpoint for run {args.run_name}'
        )
        artifact.add_file(save_path)
        wandb.log_artifact(artifact)

def generate_square_subsequent_mask(sz):
    """Generate mask for transformer to prevent positions from attending to subsequent positions.
    
    Args:
        sz (int): Size of the square mask
        
    Returns:
        Tensor: Generated mask
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def train(model, args):
    """
    Train the transformer model with wandb logging.
    
    Args:
        model: The transformer model
        args: Parsed command line arguments containing:
            - batch_size: Batch size for training
            - iterations: Number of iterations to train
            - seq_length: Length of input sequence
            - num_samples: Number of random samples to generate for training
            - rot: Rotation amount for the output text
            - learning_rate: Learning rate for optimization
            - wandb_project: WandB project name
            - wandb_entity: WandB entity name
            - run_name: WandB run name
    """
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.run_name,
        config=vars(args)
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()
    
    wandb.watch(model, log="all")
    
    dataset = BrainrotXDataset(args.seq_length, args.num_samples, rot=args.rot)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.95)
    criterion = nn.CrossEntropyLoss()
    
    src_mask = generate_square_subsequent_mask(args.seq_length).to(device)
    
    for iteration in trange(args.iterations, desc="Training Progress"):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        model.train()
        
        for batch_idx, (input_batch, target_batch) in enumerate(tqdm(dataloader, desc=f"Iteration {iteration+1}", leave=False)):
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            
            optimizer.zero_grad()
            output = model(input_batch, src_mask)
            
            output_flat = output.view(-1, 26)
            target_flat = target_batch.view(-1)
            
            loss = criterion(output_flat, target_flat)
            total_loss += loss.item()
            
            predictions = torch.argmax(output, dim=-1)
            correct_predictions += (predictions == target_batch).sum().item()
            total_predictions += target_batch.numel()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
        
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / total_predictions
        
        wandb.log({
            'train_loss': avg_loss,
            'train_accuracy': accuracy,
            'learning_rate': scheduler.get_last_lr()[0],
            'iteration': iteration + 1
        })
        
        print(f'Iteration {iteration+1}/{args.iterations}, '
              f'Average Loss: {avg_loss:.4f}, '
              f'Accuracy: {accuracy:.4f}, '
              f'LR: {scheduler.get_last_lr()[0]:.6f}')
        
    save_model(model, args, iteration)    
    wandb.finish()

def generate_output(model, text, rot=26, device=None):
    """
    Generate output from the model and show the rotation transformation.
    
    Args:
        model: The transformer model
        text: Input text string
        rot: Rotation amount (default=26)
        device: Computation device (default=None, will use CUDA if available)
    
    Returns:
        tuple: (model_output, expected_output)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if not text or not isinstance(text, str):
        raise ValueError("Input text must be a non-empty string")
    if not all(c.lower() in string.ascii_lowercase for c in text):
        raise ValueError("Input text must contain only ASCII letters")
    
    text = text.lower()
    
    model.eval()
    with torch.no_grad():
        x = torch.tensor([[ord(c) - ord('a') for c in text]], dtype=torch.long).to(device)
        src_mask = generate_square_subsequent_mask(len(text)).to(device)
        output = model(x, src_mask)
        output_indices = torch.argmax(output, dim=-1).squeeze()
        
        model_output = ''.join(chr(idx.item() + ord('a')) for idx in output_indices)
        expected_output = ''.join(
            chr(((ord(c) - ord('a') + rot) % 26) + ord('a')) 
            for c in text
        )
        
        print(f"Input text   : {text}")
        print(f"Expected     : {expected_output}")
        print(f"Model output : {model_output}")
        
        return model_output, expected_output

if __name__ == "__main__":
    args = parse_args()
    
    model = BrainrotX(
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout
    )
    print("Number of parameters:", sum(p.numel() for p in model.parameters()))
    
    print("Starting training...")
    train(model, args)
    
    test_texts = ["hello", "world", "transformer"]
    print("\nTesting model outputs:")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    for text in test_texts:
        model_out, expected = generate_output(model, text, rot=args.rot, device=device)
        print("-" * 40)