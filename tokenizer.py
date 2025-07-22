from collections import defaultdict
import json
import os
from Model_Params import end_char

# Global variables for tokenization state
merges = dict()         # Stores learned merge rules: (token1, token2) -> new_token_id
cur_vocab_size = None   # Current vocabulary size
stoi = None            # String to integer mapping: "token" -> token_id
itos = None            # Integer to string mapping: token_id -> "token"

# Helper function to display text with visible newline characters
show = lambda text: text.replace('\n', end_char)


def initialize_tokenizer(chars):
    """
    Initialize the tokenizer with character-level mappings.
    
    Sets up the initial vocabulary using individual characters.
    This forms the base vocabulary before any merges are learned.
    
    Args:
        chars (list): List of unique characters to initialize vocabulary
    """
    global stoi, itos, cur_vocab_size
    
    # Create bidirectional mappings between characters and indices
    stoi = {s: i for i, s in enumerate(chars)}  # String to integer
    itos = {i: s for s, i in stoi.items()}      # Integer to string
    cur_vocab_size = len(chars)


def can_be_merged(c1, c2):
    """
    Check if two characters/tokens can be merged according to tokenization rules.
    
    Implements constraints to ensure meaningful merges:
    - Cannot merge with end-of-line character
    - Cannot merge with special token 0 (" ")
    - Cannot merge non-space with space (prevents odd tokenization)
    
    Args:
        c1 (int): First token ID
        c2 (int): Second token ID
        
    Returns:
        bool: True if the pair can be merged, False otherwise
    """
    # Don't merge with end-of-line character
    if itos[c1] == end_char or itos[c2] == end_char:
        return False
    
    # Don't merge with special token 0
    if c1 == 0 or c2 == 0:
        return False
    
    # Don't merge non-space followed by space (prevents odd word boundaries)
    if c1 != stoi[' '] and c2 == stoi[' ']:
        return False
    
    return True


def most_often(txt):
    """
    Find the most frequent pair of adjacent tokens that can be merged.
    
    Scans through the text to count occurrences of all adjacent token pairs,
    then returns the most frequent pair that satisfies merge constraints.
    
    Args:
        txt (list): List of token IDs representing the text
        
    Returns:
        tuple: The most frequent mergeable pair (token1_id, token2_id)
    """
    # Count frequency of all adjacent pairs
    pair_counts = defaultdict(int)
    for i in range(len(txt) - 1):
        pair = (txt[i], txt[i + 1])
        if can_be_merged(pair[0], pair[1]):
            pair_counts[pair] += 1

    # Find the most frequent pair (sort by negative count for descending order)
    sorted_pairs = sorted((-count, pair) for pair, count in pair_counts.items())
    return sorted_pairs[0][1]  # Return the pair with highest count


def replace(txt, pair, new_token_id):
    """
    Replace all occurrences of a token pair with a new single token.
    
    Scans through the text and replaces every occurrence of the specified
    pair with the new token ID, effectively merging the pair.
    
    Args:
        txt (list): List of token IDs
        pair (tuple): Pair of token IDs to replace (token1_id, token2_id)
        new_token_id (int): New token ID to replace the pair with
        
    Returns:
        list: Updated list of token IDs with pairs replaced
    """
    result = []
    i = 0
    
    while i < len(txt):
        # Check if current position matches the pair to replace
        if (i + 1 < len(txt) and 
            txt[i] == pair[0] and 
            txt[i + 1] == pair[1]):
            # Replace pair with new token
            result.append(new_token_id)
            i += 2  # Skip both tokens in the pair
        else:
            # Keep current token unchanged
            result.append(txt[i])
            i += 1
            
    return result


def generate(txt, count):
    """
    Generate new tokens by iteratively merging the most frequent pairs.
    
    This is the core BPE algorithm that:
    1. Finds the most frequent mergeable pair
    2. Creates a new token for this pair
    3. Replaces all occurrences of the pair with the new token
    4. Updates vocabulary mappings and merge rules
    5. Repeats for the specified number of iterations
    
    Args:
        txt (list): List of token IDs representing the training text
        count (int): Number of merge operations to perform
        
    Returns:
        list: Updated list of token IDs after all merges
    """
    global cur_vocab_size
    
    result_text = txt
    
    for iteration in range(count):
        # Find the most frequent mergeable pair
        most_frequent_pair = most_often(result_text)
        
        # Replace all occurrences of this pair with a new token
        result_text = replace(result_text, most_frequent_pair, cur_vocab_size)
        
        # Create the merged token string
        merged_token = itos[most_frequent_pair[0]] + itos[most_frequent_pair[1]]
        
        # Update vocabulary mappings
        itos[cur_vocab_size] = merged_token
        stoi[merged_token] = cur_vocab_size
        
        # Store the merge rule
        merges[most_frequent_pair] = cur_vocab_size
        
        # Print progress information
        print(f"Zastąpiłem {most_frequent_pair} numerem {cur_vocab_size}: "
              f"'{show(itos[most_frequent_pair[0]])}' + '{show(itos[most_frequent_pair[1]])}' "
              f"-> '{show(itos[cur_vocab_size])}'")
        
        # Increment vocabulary size for next iteration
        cur_vocab_size += 1
        
    return result_text


def encode(txt):
    """
    Encode text string into a sequence of token IDs using learned merges.
    
    Starts with character-level tokenization, then iteratively applies
    learned merge rules to create longer tokens where possible.
    
    Args:
        txt (str): Input text string to encode
        
    Returns:
        list: List of token IDs representing the encoded text
    """
    # Start with character-level tokenization
    result = []
    for char in txt:
        result.append(stoi[char])
    
    # Apply learned merges iteratively
    # Continue until no more merges can be applied
    while len(result) >= 2:
        # Check if the last two tokens can be merged
        if (result[-2], result[-1]) in merges:
            # Get the merged token ID
            merged_token_id = merges[(result[-2], result[-1])]
            # Remove the two separate tokens
            result.pop()  # Remove last token
            result.pop()  # Remove second-to-last token
            # Add the merged token
            result.append(merged_token_id)
        else:
            # No merge possible, stop checking
            break
            
    return result


def decode(token_ids):
    """
    Decode a sequence of token IDs back into a text string.
    
    Args:
        token_ids (list): List of token IDs to decode
        
    Returns:
        str: Decoded text string
    """
    return ''.join(itos[token_id] for token_id in token_ids)


def decode_bar(token_ids):
    """
    Decode tokens with bar separators for debugging/visualization.
    
    Useful for understanding how text is tokenized by showing
    token boundaries explicitly.
    
    Args:
        token_ids (list): List of token IDs to decode
        
    Returns:
        str: Decoded text with '|' separating each token
    """
    return '|'.join(itos[token_id] for token_id in token_ids)


def get_vocab_size():
    """
    Get the current vocabulary size.
    
    Returns:
        int: Current number of tokens in vocabulary
    """
    return cur_vocab_size


def get_merges():
    """
    Get the current merges dictionary.
    
    Returns:
        dict: Dictionary mapping token pairs to merged token IDs
    """
    return merges


def get_mappings():
    """
    Get the string-to-int and int-to-string mappings.
    
    Returns:
        tuple: (stoi, itos) - the bidirectional vocabulary mappings
    """
    return stoi, itos


def save_tokenizer(save_dir="tokenizer_save"):
    """
    Save all tokenizer data to files for persistence.
    
    Serializes all tokenizer state (vocabularies, merges, size) to JSON files
    in the specified directory. Handles type conversions needed for JSON format.
    
    Args:
        save_dir (str): Directory path to save tokenizer files
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save string-to-integer mapping
    stoi_path = os.path.join(save_dir, "stoi.json")
    with open(stoi_path, 'w', encoding='utf-8') as f:
        json.dump(stoi, f, ensure_ascii=False, indent=2)
    
    # Save integer-to-string mapping
    # Convert int keys to strings for JSON compatibility
    itos_path = os.path.join(save_dir, "itos.json")
    itos_str_keys = {str(k): v for k, v in itos.items()}
    with open(itos_path, 'w', encoding='utf-8') as f:
        json.dump(itos_str_keys, f, ensure_ascii=False, indent=2)
    
    # Save merge rules
    # Convert tuple keys to strings for JSON compatibility
    merges_path = os.path.join(save_dir, "merges.json")
    merges_str_keys = {f"{k[0]}_{k[1]}": v for k, v in merges.items()}
    with open(merges_path, 'w', encoding='utf-8') as f:
        json.dump(merges_str_keys, f, indent=2)
    
    # Save vocabulary size
    vocab_size_path = os.path.join(save_dir, "vocab_size.json")
    with open(vocab_size_path, 'w') as f:
        json.dump({"vocab_size": cur_vocab_size}, f, indent=2)
    
    print(f"Tokenizer data saved to '{save_dir}/' directory:")
    print(f"  - stoi.json: {len(stoi)} character mappings")
    print(f"  - itos.json: {len(itos)} token mappings")
    print(f"  - merges.json: {len(merges)} merge rules")
    print(f"  - vocab_size.json: vocabulary size = {cur_vocab_size}")


def load_tokenizer(save_dir="tokenizer_save"):
    """
    Load tokenizer data from saved files.
    
    Restores all tokenizer state from JSON files, handling type conversions
    needed to restore original data structures from JSON format.
    
    Args:
        save_dir (str): Directory path containing saved tokenizer files
    """
    global stoi, itos, merges, cur_vocab_size
    
    # Load string-to-integer mapping
    stoi_path = os.path.join(save_dir, "stoi.json")
    with open(stoi_path, 'r', encoding='utf-8') as f:
        stoi = json.load(f)
    
    # Load integer-to-string mapping
    # Convert string keys back to integers
    itos_path = os.path.join(save_dir, "itos.json")
    with open(itos_path, 'r', encoding='utf-8') as f:
        itos_str_keys = json.load(f)
        itos = {int(k): v for k, v in itos_str_keys.items()}
    
    # Load merge rules
    # Convert string keys back to tuples
    merges_path = os.path.join(save_dir, "merges.json")
    with open(merges_path, 'r', encoding='utf-8') as f:
        merges_str_keys = json.load(f)
        merges = {tuple(map(int, k.split('_'))): v for k, v in merges_str_keys.items()}
    
    # Load vocabulary size
    vocab_size_path = os.path.join(save_dir, "vocab_size.json")
    with open(vocab_size_path, 'r') as f:
        vocab_data = json.load(f)
        cur_vocab_size = vocab_data["vocab_size"]
    
    print(f"Tokenizer data loaded from '{save_dir}/' directory:")
    print(f"  - Loaded {len(stoi)} character mappings")
    print(f"  - Loaded {len(itos)} token mappings") 
    print(f"  - Loaded {len(merges)} merge rules")
    print(f"  - Vocabulary size = {cur_vocab_size}")
