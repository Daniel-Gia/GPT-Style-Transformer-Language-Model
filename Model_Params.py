# Vocabulary and sequence parameters
vocab_size = 256    # Total number of tokens in vocabulary
block_size = 32     # Maximum context length (sequence length)

# Model architecture parameters
emb_len = 256       # Embedding dimension (must be divisible by num_heads)
num_layers = 6      # Number of transformer blocks
num_heads = 4       # Number of attention heads (emb_len must be divisible by this)

# Training parameters
batch_size = 16     # Number of sequences per training batch
dropout = 0.1       # Dropout probability for regularization

# Special tokens
end_char = '@'      # Character used to represent line endings