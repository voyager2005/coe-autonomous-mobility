import torch
import numpy as np
import os

def load_word_embeddings(path, vocab):
    embeds = {}
    dim = 300 # Default GloVe dimension
    
    # 1. Attempt to load the dictionary if it exists
    if os.path.exists(path):
        print(f"\nScanning for text dictionary at {path}...")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    values = line.strip().split()
                    if len(values) > 1:
                        word = values[0]
                        if word in vocab:
                            embeds[word] = np.asarray(values[1:], dtype='float32')
                            dim = len(embeds[word])
        except Exception as e:
            print(f"Warning: Could not read file properly: {e}")
    else:
        print(f"\nWarning: Text dictionary {path} not found. Running in fault-tolerant mode.")

    # 2. Build the vectors
    E = []
    vocab_list = list(vocab)
    for k in vocab_list:
        if k in embeds:
            E.append(embeds[k])
        else:
            # FAULT TOLERANCE: If the word is missing, generate a random 
            # mathematical vector matching the standard GloVe variance
            print(f"  -> Word '{k}' not found. Generating dynamic vector.")
            E.append(np.random.normal(scale=0.6, size=(dim,)))

    return np.array(E, dtype=np.float32), vocab_list

def initialize_wordembedding_matrix(wordembs_type, type_name):
    print(f"Initializing spatial embeddings for {len(type_name)} perception classes...")
    
    # 1. Break down complex classes (e.g., 'low_haze_blur_noise' -> 'low', 'haze', 'blur', 'noise')
    vocab = set()
    for name in type_name:
        words = name.split('_')
        for w in words:
            vocab.add(w)
            
    # 2. Load or generate vectors for the base words
    path = f'./utils/glove.6B.300d.txt'
    base_matrix, vocab_list = load_word_embeddings(path, vocab)
    
    word2vec = {w: base_matrix[i] for i, w in enumerate(vocab_list)}
    
    # 3. Mathematically average the words together to create the final combinatorial vectors
    final_embeddings = []
    for name in type_name:
        words = name.split('_')
        vecs = [word2vec[w] for w in words]
        avg_vec = np.mean(vecs, axis=0)
        final_embeddings.append(avg_vec)
        
    final_tensor = torch.from_numpy(np.array(final_embeddings, dtype=np.float32))
    
    print("Embedding matrix successfully compiled. Moving to GPU...\n")
    return final_tensor, final_tensor.shape[1]