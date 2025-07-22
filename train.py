import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from Model_Params import *
from tokenizer import encode, load_tokenizer
from Model import *
import glob
import os

# Clear GPU memory cache to free up space
torch.cuda.empty_cache()

# Configure device (GPU if available, otherwise CPU)
if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
    torch.set_default_device('cuda')
else:
    print("CUDA is not available. Using CPU.")
    torch.set_default_device('cpu')

# Load pre-trained tokenizer (must be created by running init.py first)
load_tokenizer()

# Load and preprocess training data
print("Loading and tokenizing training data...")
full_text = open('Train_Data/input.txt', 'r').read().replace('\n', end_char)
tok_text = encode(full_text)  # Convert text to token IDs using BPE tokenizer

# Prepare training sequences
# Create overlapping sequences of length block_size for training
print("Preparing training sequences...")
X, Y = [], []  # Input sequences and target sequences

# Generate training examples with sliding window
# Each example: input = tokens[i:i+block_size], target = tokens[i+1:i+1+block_size]
for i in range(len(tok_text) - block_size - 1):
    begin = i
    X.append(tok_text[begin:begin + block_size])        # Input sequence
    Y.append(tok_text[begin + 1:begin + 1 + block_size])  # Target sequence (shifted by 1)

# Convert to PyTorch tensors
X = torch.tensor(X)
Y = torch.tensor(Y)
print(f"Created {len(Y)} training examples")

# Split data into training and validation sets (90% train, 10% validation)
split_idx = int(len(Y) * 0.9)
X_train, X_val = X[:split_idx], X[split_idx:]
Y_train, Y_val = Y[:split_idx], Y[split_idx:]

print(f"Training examples: {len(X_train)}")
print(f"Validation examples: {len(X_val)}")

# Initialize the transformer model
T = Transformer(block_size, emb_len, vocab_size, num_layers)

# Count and display model parameters
parameters = list(T.parameters())
total_params = sum(p.numel() for p in parameters if p.requires_grad)
print(f"Model initialized with {total_params:,} trainable parameters")

# Load existing model weights if available
weight_files = sorted(glob.glob(os.path.join("Saved_Weights", "weights-*.pth")))
if weight_files:
    # Load the most recently created weight file
    latest_weight = max(weight_files, key=os.path.getctime)
    print(f"Loading model weights from {latest_weight}")
    T.load_state_dict(torch.load(latest_weight))
else:
    print("No saved weights found. Training from scratch.")


def calc_val_loss():
    """
    Calculate validation loss on the entire validation set.
    
    Uses torch.no_grad() to disable gradient computation for efficiency.
    This helps monitor overfitting during training.
    
    Returns:
        float: Average validation loss
    """
    with torch.no_grad():
        _, loss = T(X_val, Y_val)
    return loss.item()


# Initialize optimizer
# Adam optimizer with learning rate 1e-3 (standard for transformer training)
optimizer = torch.optim.Adam(T.parameters(), lr=1e-3)

# Training configuration
show_it = 1000    # Print progress every 1000 iterations
save_it = 10000   # Save model weights every 10000 iterations

# Training loop tracking
XS, YS = [], []   # Store iteration numbers and loss values for plotting

print("Starting training...")
print(f"Will show progress every {show_it} iterations")
print(f"Will save model every {save_it} iterations")

# Main training loop
total_iterations = 1000 * 10 * 5  # 50,000 iterations total

for iteration in range(total_iterations):
    # Sample random batch from training data
    batch_indices = torch.randint(0, len(Y_train), (batch_size,))
    
    # FORWARD PASS
    # Get predictions and calculate loss
    logits, loss = T(X_train[batch_indices], Y_train[batch_indices])

    # BACKWARD PASS
    # Clear gradients from previous iteration
    optimizer.zero_grad()
    
    # Compute gradients
    loss.backward()
    
    # UPDATE PARAMETERS
    # Apply gradients to update model weights
    optimizer.step()
    
    # Progress monitoring
    if iteration % show_it == 0:
        val_loss = calc_val_loss()
        print(f"Iteration {iteration:6d} | "
              f"Train Loss: {loss.item():.4f} | "
              f"Val Loss: {val_loss:.4f}")
    
    # Store loss for plotting
    XS.append(iteration)
    YS.append(loss.item())

    # Save model checkpoint
    if iteration % save_it == save_it - 1:
        checkpoint_path = os.path.join("Saved_Weights", f"weights-{iteration//save_it:02d}.pth")
        print(f"Saving model checkpoint to {checkpoint_path}")
        torch.save(T.state_dict(), checkpoint_path)

print("Training completed!")

# Plot training loss curve
print("Generating loss plot...")
plt.figure(figsize=(10, 6))
plt.plot(XS, YS)
plt.title('Training Loss Over Time')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

print("Training script finished. Model weights saved to 'Saved_Weights/' directory.")