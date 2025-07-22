"""
Text Generation Script

This script loads a trained transformer model and generates text continuations
based on user input. It demonstrates the model's ability to generate coherent
text by predicting the next token iteratively.

Features:
- Loads the most recent saved model weights
- Interactive text input from user
- Autoregressive text generation (one token at a time)
- Multinomial sampling for diverse outputs
- Automatic stopping at end-of-line markers

The generation process:
1. Encode user input to token sequence
2. Use model to predict next token probabilities
3. Sample from probability distribution
4. Append sampled token and repeat
5. Stop when end token is generated
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from Model_Params import *
from tokenizer import encode, load_tokenizer, decode
from Model import *
import glob
import os
import sys

# Clear GPU memory cache to free up space
torch.cuda.empty_cache()

# Configure device (GPU if available, otherwise CPU)
if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
    torch.set_default_device('cuda')
else:
    print("CUDA is not available. Using CPU.")
    torch.set_default_device('cpu')

# Initialize the transformer model with saved architecture
print("Initializing transformer model...")
T = Transformer(block_size, emb_len, vocab_size, num_layers)

# Load the most recent model weights
print("Looking for saved model weights...")
weight_files = sorted(glob.glob(os.path.join("Saved_Weights", "weights-*.pth")))

if weight_files:
    # Find and load the most recently created weight file
    latest_weight = max(weight_files, key=os.path.getctime)
    print(f"Loading model weights from {latest_weight}")
    T.load_state_dict(torch.load(latest_weight))
    print("Model loaded successfully!")
else:
    print("Error: No saved weights found in 'Saved_Weights/' directory.")
    print("Please train the model first by running train.py")
    sys.exit()

# Load the tokenizer (vocabulary and merge rules)
print("Loading tokenizer...")
load_tokenizer()

# Get input text from user
print("\n" + "="*50)
print("Text Generation with Transformer Model")
print("="*50)
input_text = input("Enter text to generate continuation: ")

# Validate input
if not input_text:
    print("No input text provided. Exiting.")
    sys.exit()

# Encode user input to token sequence
print(f"\nInput text: '{input_text}'")
int_text = encode(input_text)
print(f"Encoded to {len(int_text)} tokens: {int_text}")

# Handle input that's too long for the model's context window
if len(int_text) > block_size:
    print(f"Warning: Input text is too long ({len(int_text)} tokens).")
    print(f"Truncating to last {block_size} tokens to fit model's context window.")
    int_text = int_text[-block_size:]

# Get the end token ID for stopping generation
end_token = encode("@")[0]  # The special end-of-line character
print(f"End token ID: {end_token}")

print("\nGenerating text...")
print("-" * 30)

# Text generation loop
generation_steps = 0
max_generation_steps = 200  # Prevent infinite generation

while generation_steps < max_generation_steps:
    # Prepare input context (last block_size tokens)
    context = int_text[-block_size:]
    
    # Convert to tensor and add batch dimension
    context_tensor = torch.tensor(context).view(1, -1)
    
    # Forward pass through the model
    with torch.no_grad():  # Disable gradient computation for inference
        logits, _ = T(context_tensor)
    
    # Get logits for the last position (next token prediction)
    logits = logits[:, -1, :]  # Shape: (1, vocab_size)
    
    # Convert logits to probabilities using softmax
    probabilities = F.softmax(logits, dim=-1)
    
    # Sample next token from probability distribution
    # Using multinomial sampling for diversity (not always picking highest probability)
    next_token_id = torch.multinomial(probabilities, num_samples=1).item()
    
    # Add the generated token to our sequence
    int_text.append(next_token_id)
    generation_steps += 1
    
    # Check if we generated the end token (stop generation)
    if next_token_id == end_token:
        print("Generated end token. Stopping generation.")
        break
    
    # Optional: Print each generated token for real-time monitoring
    # print(f"Step {generation_steps}: Generated token {next_token_id}")

if generation_steps >= max_generation_steps:
    print(f"Reached maximum generation steps ({max_generation_steps}). Stopping.")

# Display results
print("\n" + "="*50)
print("GENERATION RESULTS")
print("="*50)

print(f"\nGenerated sequence (token IDs):")
print(int_text)

print(f"\nGenerated text:")
generated_text = decode(int_text)
print(f"'{generated_text}'")

print(f"\nGeneration statistics:")
print(f"- Total tokens generated: {generation_steps}")
print(f"- Final sequence length: {len(int_text)} tokens")
print(f"- Input length: {len(encode(input_text))} tokens")
print(f"- Generated length: {generation_steps} tokens")

# Replace end character with actual newlines for better readability
readable_text = generated_text.replace(end_char, '\n')
print(f"\nReadable output:")
print("-" * 30)
print(readable_text)
print("-" * 30)

print("\nText generation completed!")
