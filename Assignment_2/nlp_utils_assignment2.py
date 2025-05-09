"""
NLP Utilities for Assignment 2: Embeddings, Visualization, and Nearest Neighbors
"""

import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import gensim.downloader as api

# Ensure nltk resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading punkt tokenizer...")
    nltk.download('punkt')

# Define a small custom corpus - mix of various topics to ensure interesting relationships
custom_corpus = [
    "The king and queen ruled the kingdom with fairness and wisdom.",
    "The president addressed the nation about economic policies.",
    "The cat and dog played in the garden all afternoon.",
    "Machine learning models need large datasets for training.",
    "The knight defended the castle against invaders.",
    "Python is a popular programming language for data science.",
    "The apple tree produced delicious fruit this season.",
    "Neural networks can learn complex patterns from data.",
    "The king ordered his knights to protect the royal palace.",
    "The queen hosted a grand ball for nobles from across the land.",
    "Dogs are often called man's best friend due to their loyalty.",
    "Algorithms form the foundation of computer science.",
    "The castle walls stood tall against the enemy forces.",
    "Natural language processing helps computers understand human language.",
    "The fruit market had apples, oranges, and bananas for sale.",
]

def generate_tf_idf_embeddings(corpus=None):
    """
    Generates TF-IDF embeddings for a given corpus.
    If no corpus is provided, uses the default custom corpus.
    """
    if corpus is None:
        corpus = custom_corpus
        
    # Tokenize and normalize corpus (optional preprocessing)
    # processed_corpus = [" ".join(nltk.word_tokenize(doc.lower())) for doc in corpus]
    
    # Create TF-IDF vectorizer and fit it to the corpus
    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    return vectorizer, tfidf_matrix

def get_word_tfidf_embedding(word, vectorizer, tfidf_matrix=None):
    """
    Gets the TF-IDF representation for a specific word.
    This is a simplification - TF-IDF is normally applied to documents, not words.
    Here we either:
    1. Return the column vector corresponding to this word in the vocabulary, or
    2. Create a single-word document and get its TF-IDF vector
    """
    # Check if word is in vocabulary
    if word in vectorizer.vocabulary_:
        # Option 1: Return the weight of this word across all documents
        word_idx = vectorizer.vocabulary_[word]
        if tfidf_matrix is not None:
            # Get column for this word from the document-term matrix
            return tfidf_matrix[:, word_idx].toarray().flatten()
        else:
            # Without a matrix, we'll create a single-word document
            pass
    
    # Option 2: Create a document with just this word and get its vector
    word_doc = [word]
    word_vec = vectorizer.transform(word_doc)
    return word_vec.toarray().flatten()

def load_glove_model(model_name='glove-wiki-gigaword-50'):
    """
    Loads a pre-trained GloVe model from gensim.
    Default is 'glove-wiki-gigaword-50' (50-dim GloVe vectors trained on Wikipedia & Gigaword).
    """
    print(f"Loading {model_name}...")
    model = api.load(model_name)
    print(f"{model_name} loaded successfully.")
    return model

def generate_glove_embeddings(words, model):
    """
    Generates GloVe embeddings for a list of words using a pre-trained model.
    Returns a dictionary of {word: embedding_vector} pairs.
    """
    embeddings = {}
    for word in words:
        try:
            embeddings[word] = model[word]
        except KeyError:
            # For unknown words, return a zero vector of the same dimension
            embeddings[word] = np.zeros(model.vector_size)
    return embeddings

def reduce_dimensionality(embeddings_matrix, method='tsne', n_components=2):
    """
    Reduces dimensionality of embeddings using t-SNE or PCA.
    
    Args:
        embeddings_matrix: Matrix where each row is an embedding vector
        method: 'tsne' or 'pca'
        n_components: Number of dimensions to reduce to (usually 2 for visualization)
        
    Returns:
        Reduced matrix with shape (n_samples, n_components)
    """
    if method == 'tsne':
        # Adjust perplexity based on data size to avoid warnings
        perplexity = min(30, max(5, embeddings_matrix.shape[0] // 5))
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=perplexity)
    elif method == 'pca':
        reducer = PCA(n_components=n_components, random_state=42)
    else:
        raise ValueError("Unsupported dimensionality reduction method. Choose 'tsne' or 'pca'.")
    
    reduced_embeddings = reducer.fit_transform(embeddings_matrix)
    return reduced_embeddings

def find_nearest_neighbors(word_embedding, all_embeddings, word_list, n_neighbors=5):
    """
    Finds nearest neighbors for a given word embedding using cosine similarity.
    
    Args:
        word_embedding: The embedding vector of the target word
        all_embeddings: Dictionary of {word: embedding_vector} pairs
        word_list: List of words to consider as potential neighbors
        n_neighbors: Number of neighbors to return
        
    Returns:
        List of (word, similarity_score) pairs for the nearest neighbors
    """
    if not isinstance(all_embeddings, dict) or not word_list:
        return []
    
    # Reshape the target word's embedding to a 2D array for cosine_similarity
    target_word_vec = word_embedding.reshape(1, -1)
    
    # Filter valid words and their embeddings
    valid_words = []
    embedding_vectors = []
    
    for w in word_list:
        if w in all_embeddings:
            valid_words.append(w)
            embedding_vectors.append(all_embeddings[w])
    
    if not embedding_vectors:
        return []
    
    # Convert list of embeddings to a 2D numpy array
    embedding_matrix = np.array(embedding_vectors)
    
    # Calculate cosine similarities between the target word and all other words
    similarities = cosine_similarity(target_word_vec, embedding_matrix).flatten()
    
    # Get indices of words sorted by similarity (descending)
    sorted_indices = np.argsort(similarities)[::-1]
    
    # Collect nearest neighbors (excluding the word itself)
    neighbors = []
    target_word = word_list[0]  # Assuming the first word in word_list is the target word
    
    for i in sorted_indices:
        if valid_words[i] != target_word and len(neighbors) < n_neighbors:
            neighbors.append((valid_words[i], float(similarities[i])))
    
    return neighbors

def extract_words_from_corpus(corpus=None):
    """
    Extracts unique words from the corpus for use in embeddings and visualizations.
    """
    if corpus is None:
        corpus = custom_corpus
        
    all_words = set()
    for doc in corpus:
        tokens = nltk.word_tokenize(doc.lower())
        all_words.update(tokens)
    
    # Remove punctuation and common stop words (simplified approach)
    stop_words = set(['the', 'and', 'a', 'in', 'from', 'to', 'for', 'of', 'with', 'by', '.', ',', '?', '!'])
    filtered_words = [word for word in all_words if word not in stop_words and word.isalpha()]
    
    return filtered_words

# Example usage (will be part of the API or web app logic)
if __name__ == '__main__':
    # TF-IDF example
    vectorizer, tfidf_matrix = generate_tf_idf_embeddings()
    print("TF-IDF Vocabulary size:", len(vectorizer.get_feature_names_out()))
    
    # Get TF-IDF embedding for a word
    word = "king"
    word_tfidf = get_word_tfidf_embedding(word, vectorizer, tfidf_matrix)
    print(f"TF-IDF representation for '{word}':", word_tfidf[:5], "...")
    
    # GloVe example
    try:
        model = load_glove_model()
        
        # Extract words from our corpus
        corpus_words = extract_words_from_corpus()
        print(f"Extracted {len(corpus_words)} unique words from corpus")
        
        # Get embeddings for some sample words
        sample_words = ["king", "queen", "castle", "knight", "python", "computer"]
        sample_words = [w for w in sample_words if w in corpus_words]
        
        glove_embeddings = generate_glove_embeddings(sample_words, model)
        print("\nGloVe Embeddings (first 5 dimensions):")
        for word, embedding in glove_embeddings.items():
            print(f"{word}: {embedding[:5]}...")
            
        # Example: Convert embeddings to a format suitable for dimensionality reduction
        words_for_viz = list(glove_embeddings.keys())
        embedding_vectors = np.array([glove_embeddings[w] for w in words_for_viz])
        
        # Reduce dimensions for visualization
        reduced_vectors_tsne = reduce_dimensionality(embedding_vectors, method='tsne')
        reduced_vectors_pca = reduce_dimensionality(embedding_vectors, method='pca')
        
        print("\nReduced dimensions (t-SNE):")
        for i, word in enumerate(words_for_viz):
            print(f"{word}: ({reduced_vectors_tsne[i, 0]:.4f}, {reduced_vectors_tsne[i, 1]:.4f})")
            
        # Find nearest neighbors for 'king'
        if "king" in glove_embeddings:
            king_embedding = glove_embeddings["king"]
            neighbors = find_nearest_neighbors(king_embedding, glove_embeddings, words_for_viz)
            print(f"\nNearest neighbors for 'king':")
            for word, similarity in neighbors:
                print(f"{word}: {similarity:.4f}")
                
    except Exception as e:
        print(f"Error in GloVe example: {str(e)}") 