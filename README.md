# GPT-Style Transformer Language Model

## 🎯 What This Project Is / Does

This project is a **complete implementation of a GPT-style transformer language model from scratch** using PyTorch. It demonstrates the fundamental architecture and training procedures used in modern large language models like GPT, ChatGPT, and similar systems.

### Key Features:
- **Custom Byte Pair Encoding (BPE) Tokenizer**: Implements subword tokenization from scratch
- **Transformer Architecture**: Multi-head self-attention, feed-forward networks, layer normalization
- **Training Pipeline**: Complete training loop with validation monitoring and checkpointing  
- **Text Generation**: Autoregressive text generation with sampling

## 🎓 Educational Purpose & Limitations

**This project is designed to demonstrate understanding of LLM/Attention Architecture fundamentals.**

⚠️ **Important Note**: The pretrained model included is just a **small example** and is **not great** for practical use. It's trained on a limited dataset with minimal parameters to showcase the architecture and training process.

**If you want better results:**
- Increase model parameters (embedding dimension, layers, heads)
- Train on larger, more diverse datasets
- Extend training time (more epochs/iterations)
- Tune hyperparameters (learning rate, batch size, etc.)

**For Training Larger Models:**
If you want to train bigger, more powerful models, I recommend using **Azure ML Studio** for cloud-based training with powerful GPUs. This is especially useful for:
- Models with millions/billions of parameters
- Large datasets that don't fit in local memory
- Faster training with high-end GPU clusters

📺 **Helpful Tutorial**: [Azure ML Studio Training Guide](https://www.youtube.com/watch?v=L16GXVOnqCM&list=WL) - Great tutorial for getting started with cloud-based model training.

This implementation prioritizes **educational clarity** over performance optimization.

## 🚀 How to Run

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional, but recommended for faster training)

### Installation
1. Clone or download this repository
2. Install required packages:
```bash
pip install -r requirements.txt
```
> **Note:** You may need to run this command with administrator privileges (e.g., use `sudo` on Linux/macOS or run your terminal as Administrator on Windows) if you encounter permission errors.

### Quick Start (Using Pretrained Model)
If you want to use the existing trained model for text generation:

```bash
python run.py
```
Enter your text prompt when asked, and the model will generate a continuation.

### Training From Scratch
If you want to train the model yourself:

1. **Initialize the tokenizer** (creates vocabulary and merge rules):
```bash
python init.py
```

2. **Train the model** (this will take time, especially on CPU):
```bash
python train.py
```

3. **Generate text** with your trained model:
```bash
python run.py
```

### Training Data
The model is trained on Shakespeare text (`Train_Data/input.txt`). You can replace this with your own text data to train on different content.

## 📁 Project Structure

```
llm-example/
├── 📄 README.md                   # This file
├── 📄 requirements.txt            # Python package dependencies
├── 📄 Model_Params.py            # Configuration parameters and hyperparameters
├── 📄 Model.py                   # Transformer architecture implementation
├── 📄 tokenizer.py               # BPE tokenizer implementation
├── 📄 init.py                    # Tokenizer initialization script
├── 📄 train.py                   # Model training script
├── 📄 run.py                     # Text generation/inference script
├── 📁 Train_Data/                # Training data directory
│   └── 📄 input.txt              # Shakespeare text for training
├── 📁 Saved_Weights/             # Model checkpoints directory
│   └── 📄 weights-*.pth          # Saved model weights
└── 📁 tokenizer_save/            # Tokenizer data directory
    ├── 📄 stoi.json              # String-to-integer mappings
    ├── 📄 itos.json              # Integer-to-string mappings  
    ├── 📄 merges.json            # BPE merge rules
    └── 📄 vocab_size.json        # Vocabulary size information
```

### File Descriptions

#### Core Scripts
- **`Model_Params.py`**: Contains all hyperparameters and configuration settings
- **`Model.py`**: Implements the complete transformer architecture (attention, MLP, blocks)
- **`tokenizer.py`**: BPE tokenizer with encoding/decoding functionality
- **`init.py`**: Sets up and trains the tokenizer on your text data
- **`train.py`**: Main training loop with validation and checkpointing
- **`run.py`**: Interactive text generation using trained model

#### Data Directories
- **`Train_Data/`**: Contains training text data
- **`Saved_Weights/`**: Stores model checkpoints during training
- **`tokenizer_save/`**: Stores tokenizer vocabulary and merge rules

## 🔧 Customization

### Model Architecture
Edit `Model_Params.py` to modify:
- `vocab_size`: Vocabulary size (affects tokenizer)
- `block_size`: Maximum sequence length
- `emb_len`: Embedding dimension (larger = more parameters)
- `num_layers`: Number of transformer layers
- `num_heads`: Number of attention heads
- `batch_size`: Training batch size
- `dropout`: Regularization strength

### Training Data
Replace `Train_Data/input.txt` with your own text data. The model will learn to generate text in the style of your training data.

## 🧠 Technical Details

### Architecture Components
1. **Token & Positional Embeddings**: Convert tokens to dense vectors with position information
2. **Multi-Head Self-Attention**: Allows model to attend to different parts of the sequence
3. **Feed-Forward Networks**: Process attended information through MLPs
4. **Layer Normalization**: Stabilizes training and improves performance
5. **Residual Connections**: Enable training of deeper networks

### Training Process
1. **Tokenization**: Text is converted to integer sequences using BPE
2. **Batch Creation**: Sequences are batched for parallel processing
3. **Forward Pass**: Model predicts next token probabilities
4. **Loss Calculation**: Cross-entropy loss between predictions and targets
5. **Backpropagation**: Gradients are computed and weights updated
6. **Validation**: Model performance is monitored on held-out data

## 📊 Model Performance

This implementation uses a small model configuration:
- **Parameters**: ~4M parameters
- **Training Data**: Shakespeare corpus (~1MB text)
- **Performance**: Basic text generation capabilities

For comparison, GPT-3 has 175B parameters and is trained on hundreds of gigabytes of text data.

## 🤝 Contributing

This is an educational project. Feel free to:
- Experiment with different architectures
- Try different training data
- Implement additional features (beam search, top-k sampling, etc.)
- Optimize for better performance

## 📚 References

This implementation is inspired by:
- "Attention Is All You Need" (Vaswani et al., 2017)

---

**Happy learning and experimenting with transformer architectures! 🚀**
