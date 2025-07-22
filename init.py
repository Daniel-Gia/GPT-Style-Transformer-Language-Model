from Model_Params import *

# Read training data and replace newlines with special end character
full_text = open('Train_Data/input.txt', 'r').read().replace('\n', end_char)

# Extract unique characters and create initial character-to-index mappings
chars = sorted(list(set(full_text)))  # Get sorted unique characters
stoi = {s:i for i,s in enumerate(chars)}  # String to integer mapping
itos = {i:s for s,i in stoi.items()}  # Integer to string mapping

# Ensure the end character is in our vocabulary
assert end_char in stoi
chars_size = len(chars)

# Helper function to display text with visible newlines
show = lambda text: text.replace(end_char, '\n')

# Convert text to integer representation using character mappings
int_text = [stoi[c] for c in full_text]

from tokenizer import initialize_tokenizer, generate, get_vocab_size, save_tokenizer

# Initialize tokenizer with character mappings
initialize_tokenizer(chars)

# Generate new tokens
generate(int_text, vocab_size - chars_size)

# Verify we reached the target vocabulary size
assert get_vocab_size() == vocab_size

# Save all tokenizer data for later use
save_tokenizer()