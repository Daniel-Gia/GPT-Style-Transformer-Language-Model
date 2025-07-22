"""
Transformer Language Model Implementation

This module implements a complete GPT-style transformer model for language generation.
The architecture consists of:
1. Multi-head self-attention mechanism
2. Position-wise feed-forward networks (MLP)
3. Layer normalization and residual connections
4. Positional and token embeddings

Classes:
- MultiheadAttention: Implements scaled dot-product attention with multiple heads
- MLP: Feed-forward network with ReLU activation
- Block: Single transformer block combining attention and MLP with residuals
- Transformer: Complete model combining all components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from Model_Params import num_heads, dropout


class MultiheadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    
    Implements the attention mechanism from "Attention Is All You Need" paper.
    Uses causal masking to ensure autoregressive property (can only attend to previous tokens).
    
    Args:
        context_len: Maximum sequence length for positional encoding
        embed_len: Dimension of input embeddings
        num_heads: Number of parallel attention heads
        dropout_rate: Dropout probability for attention weights
    """
    
    def __init__(self, context_len, embed_len, num_heads, dropout_rate=dropout):
        super().__init__()
        self.context_len = context_len
        self.embed_len = embed_len
        self.num_heads = num_heads
        
        # Ensure embedding dimension is divisible by number of heads
        assert embed_len % num_heads == 0
        self.head_len = embed_len // num_heads

        # Linear projections for queries, keys, and values
        self.query = nn.Linear(embed_len, embed_len, bias=False)
        self.key = nn.Linear(embed_len, embed_len, bias=False)
        self.value = nn.Linear(embed_len, embed_len, bias=False)

        # Causal mask: upper triangular matrix to prevent attending to future tokens
        self.register_buffer('mask', ~torch.tril(torch.ones(context_len, context_len)).bool())
        
        # Output projection and dropout
        self.lin = nn.Linear(embed_len, embed_len)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Forward pass of multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_len)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_len)
        """
        B, T, E = x.size()  # Batch size, sequence length, embedding dimension
        
        # Generate queries, keys, and values
        q = self.query(x)  # (B, T, E)
        k = self.key(x)    # (B, T, E)
        v = self.value(x)  # (B, T, E)
        
        # Reshape for multi-head attention: (B, T, E) -> (B, H, T, E/H)
        q = q.view(B, T, self.num_heads, self.head_len).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_len).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_len).transpose(1, 2)
        
        # Compute scaled dot-product attention
        sim = (q @ k.transpose(-2, -1)) / (self.head_len ** 0.5)  # (B, H, T, T)
        
        # Apply causal mask (prevent attending to future tokens)
        sim = sim.masked_fill(self.mask[:T, :T], float('-inf'))
        
        # Apply softmax to get attention weights
        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention weights to values
        output = attn @ v  # (B, H, T, E/H)
        
        # Concatenate all heads back together: (B, H, T, E/H) -> (B, T, E)
        output = output.transpose(1, 2).contiguous().view(B, T, E)
        
        # Final linear projection
        return self.lin(output)


class MLP(nn.Module):
    """
    Multi-layer perceptron (feed-forward network).
    
    Implements the position-wise feed-forward network used in transformer blocks.
    Uses ReLU activation and includes dropout for regularization.
    
    Args:
        embed_len: Input and output dimension
        dropout_rate: Dropout probability
    """
    
    def __init__(self, embed_len, dropout_rate=dropout):
        super().__init__()
        # Standard transformer MLP: expand by factor of 4, then project back
        self.layer_1 = nn.Linear(embed_len, 4 * embed_len)
        self.relu = nn.ReLU()
        self.layer_2 = nn.Linear(4 * embed_len, embed_len)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        """
        Forward pass of the MLP.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_len)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_len)
        """
        x = self.layer_1(x)     # Expand dimension
        x = self.relu(x)        # Non-linear activation
        x = self.dropout(x)     # Regularization
        x = self.layer_2(x)     # Project back to original dimension
        return x


class Block(nn.Module):
    """
    Single transformer block.
    
    Implements one layer of the transformer architecture with:
    1. Multi-head self-attention with residual connection
    2. Feed-forward network with residual connection
    3. Layer normalization applied before each sub-layer (pre-norm)
    
    Args:
        context_len: Maximum sequence length
        embed_len: Embedding dimension
        dropout_rate: Dropout probability
    """
    
    def __init__(self, context_len, embed_len, dropout_rate=dropout):
        super().__init__()
        # Layer normalization (applied before attention and MLP)
        self.ln_1 = nn.LayerNorm(embed_len)
        self.attention = MultiheadAttention(context_len, embed_len, num_heads, dropout_rate)
        
        self.ln_2 = nn.LayerNorm(embed_len)
        self.mlp = MLP(embed_len, dropout_rate)
        
        # Dropout for residual connections
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        """
        Forward pass of the transformer block.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_len)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_len)
        """
        # Attention sub-layer with residual connection (pre-norm)
        x = x + self.dropout(self.attention(self.ln_1(x)))
        
        # MLP sub-layer with residual connection (pre-norm)
        x = x + self.dropout(self.mlp(self.ln_2(x)))
        
        return x


class Transformer(nn.Module):
    """
    Complete GPT-style transformer model for language generation.
    
    Combines token embeddings, positional embeddings, multiple transformer blocks,
    and a final linear layer for vocabulary prediction.
    
    Args:
        context_len: Maximum sequence length
        embed_len: Embedding dimension
        vocab_size: Size of vocabulary
        num_layers: Number of transformer blocks
        dropout_rate: Dropout probability
    """
    
    def __init__(self, context_len, embed_len, vocab_size, num_layers, dropout_rate=dropout):
        super().__init__()
        # Embedding layers
        self.inp_emb = nn.Embedding(vocab_size, embed_len)    # Token embeddings
        self.pos_emb = nn.Embedding(context_len, embed_len)   # Positional embeddings
        
        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            Block(context_len, embed_len, dropout_rate) 
            for _ in range(num_layers)
        ])
        
        # Final layer normalization and output projection
        self.ln_final = nn.LayerNorm(embed_len)
        self.last = nn.Linear(embed_len, vocab_size)  # Project to vocabulary size
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, targets=None):
        """
        Forward pass of the transformer model.
        
        Args:
            x: Input token indices of shape (batch_size, seq_len)
            targets: Target token indices for loss calculation (optional)
            
        Returns:
            logits: Output logits of shape (batch_size, seq_len, vocab_size)
            loss: Cross-entropy loss if targets provided, None otherwise
        """
        B, T = x.size()  # Batch size, sequence length
        device = x.device
        
        # Generate positional indices
        pos = torch.arange(T, device=device)
        
        # Combine token and positional embeddings
        x = self.dropout(self.inp_emb(x) + self.pos_emb(pos))  # (B, T, E)
        
        # Pass through all transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer normalization and output projection
        x = self.ln_final(x)
        logits = self.last(x)  # (B, T, vocab_size)
        
        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            # Flatten for cross-entropy calculation
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss