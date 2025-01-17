import argparse
import string
import torch
from model import BrainrotX
from train import generate_output

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Use a trained transformer model for text rotation')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the model checkpoint (.pt file)')
    parser.add_argument('--text', type=str, required=True,
                       help='Input text to transform')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default=None,
                       help='Device to run inference on (default: auto-detect)')
    
    return parser.parse_args()

def load_model(checkpoint_path, device):
    """Load model from checkpoint and move to specified device.
    
    Args:
        checkpoint_path (str): Path to the model checkpoint
        device (torch.device): Device to load the model onto
        
    Returns:
        tuple: (model, config) where config contains the model's training parameters
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {checkpoint_path}: {str(e)}")
    
    if 'config' not in checkpoint:
        raise ValueError("Checkpoint does not contain model configuration")
    
    config = checkpoint['config']
    required_params = ['d_model', 'nhead', 'num_layers', 'dim_feedforward', 'dropout', 'rot']
    missing_params = [param for param in required_params if param not in config]
    if missing_params:
        raise ValueError(f"Checkpoint configuration missing required parameters: {missing_params}")
    
    model = BrainrotX(
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, config

def validate_input(text):
    """Validate that input text contains only ASCII letters.
    
    Args:
        text (str): Input text to validate
        
    Returns:
        str: Lowercase version of input text if valid
        
    Raises:
        ValueError: If input text contains non-ASCII letters
    """
    if not text or not isinstance(text, str):
        raise ValueError("Input text must be a non-empty string")
    
    text = text.lower()
    invalid_chars = set(c for c in text if c not in string.ascii_lowercase)
    if invalid_chars:
        raise ValueError(
            f"Input text contains invalid characters: {invalid_chars}\n"
            "Only ASCII letters are allowed"
        )
    
    return text

def main():
    args = parse_args()
    
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        print(f"Loading model from {args.model_path}")
        model, config = load_model(args.model_path, device)
        print("Model loaded successfully")
        print(f"Model configuration: {config}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return 1
    
    try:
        text = validate_input(args.text)
    except ValueError as e:
        print(f"Error: {str(e)}")
        return 1
    
    try:
        model_output, expected_output = generate_output(
            model, 
            text, 
            rot=config['rot'],
            device=device
        )
        
        accuracy = sum(1 for a, b in zip(model_output, expected_output) if a == b) / len(text)
        print(f"\nAccuracy: {accuracy:.2%}")
        
        return 0
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return 1

if __name__ == "__main__":
    main()