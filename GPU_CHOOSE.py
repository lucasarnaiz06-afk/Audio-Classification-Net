import torch

    
# Check if MPS (Apple Silicon GPU) is available
print(f"MPS available: {torch.backends.mps.is_available()}")

# Set device automatically
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")  # For M1/M2/M3 Macs
else:
    device = torch.device("cpu")
    
print(f"Using device: {device}")
