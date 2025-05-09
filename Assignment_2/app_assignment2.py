"""
Flask App for Assignment 2: Embeddings API and Web App
"""

from flask import Flask, request, jsonify, render_template
from nlp_utils_assignment2 import (
    generate_tf_idf_embeddings,
    get_word_tfidf_embedding,
    load_glove_model,
    generate_glove_embeddings,
    reduce_dimensionality,
    find_nearest_neighbors,
    extract_words_from_corpus,
    custom_corpus
)
import numpy as np
import threading
import time

app = Flask(__name__, template_folder='templates', static_folder='static')

# Global variables for models and data
GLOVE_MODEL = None
TFIDF_VECTORIZER = None
TFIDF_MATRIX = None
CORPUS_WORDS = None
MODEL_LOADING = False
MODELS_LOADED = False

def load_models_background():
    """
    Load models in the background to avoid blocking the app startup
    """
    global GLOVE_MODEL, TFIDF_VECTORIZER, TFIDF_MATRIX, CORPUS_WORDS, MODEL_LOADING, MODELS_LOADED
    
    try:
        MODEL_LOADING = True
        
        # First, generate TF-IDF (this is quick)
        print("Generating TF-IDF vectors for corpus...")
        TFIDF_VECTORIZER, TFIDF_MATRIX = generate_tf_idf_embeddings()
        print("TF-IDF vectors generated.")
        
        # Extract words from our corpus for embeddings and comparison
        CORPUS_WORDS = extract_words_from_corpus()
        print(f"Extracted {len(CORPUS_WORDS)} unique words from corpus")
        
        # Load GloVe model (this can take time)
        try:
            print("Loading GloVe model...")
            GLOVE_MODEL = load_glove_model('glove-wiki-gigaword-50')
            print("GloVe model loaded successfully.")
        except Exception as e:
            print(f"Error loading GloVe model: {str(e)}")
            
        MODELS_LOADED = True
    except Exception as e:
        print(f"Error in model loading: {str(e)}")
    finally:
        MODEL_LOADING = False

@app.route('/')
def index():
    return render_template('index_assignment2.html')

@app.route('/api/embedding', methods=['POST'])
def get_embedding():
    data = request.get_json()
    word = data.get('word')
    embedding_type = data.get('type', 'glove')  # Default to glove

    if not word:
        return jsonify({"error": "Word not provided"}), 400

    # Check if models are loaded
    if MODEL_LOADING:
        return jsonify({
            "word": word,
            "embedding": None,
            "error": "Models are still loading. Please try again in a moment."
        }), 503  # Service Unavailable

    if not MODELS_LOADED:
        return jsonify({
            "word": word,
            "embedding": None, 
            "error": "Models have not been loaded yet. Try again later."
        }), 503  # Service Unavailable

    embedding_vector = None
    error_message = None

    try:
        if embedding_type == 'glove':
            if GLOVE_MODEL and word.lower() in GLOVE_MODEL:
                embedding_vector = GLOVE_MODEL[word.lower()].tolist()
            else:
                error_message = f"Word '{word}' not found in GloVe model."
        elif embedding_type == 'tfidf':
            if TFIDF_VECTORIZER:
                embedding_vector = get_word_tfidf_embedding(word.lower(), TFIDF_VECTORIZER, TFIDF_MATRIX).tolist()
                # If the vector is all zeros, the word isn't in our corpus
                if np.all(np.array(embedding_vector) == 0):
                    error_message = f"Word '{word}' not found in TF-IDF vocabulary or has no weight in the corpus."
            else:
                error_message = "TF-IDF model not loaded."
        else:
            error_message = f"Unsupported embedding type: {embedding_type}"

    except Exception as e:
        error_message = str(e)

    if error_message:
        return jsonify({
            "word": word,
            "embedding": None,
            "error": error_message
        }), 404 if "not found" in error_message else 500

    return jsonify({
        "word": word,
        "embedding": embedding_vector
    })

@app.route('/api/nearest_neighbors', methods=['POST'])
def get_nearest_neighbors():
    data = request.get_json()
    word = data.get('word')
    embedding_type = data.get('type', 'glove')
    n_neighbors = int(data.get('n', 5))

    if not word:
        return jsonify({"error": "Word not provided"}), 400

    # Check if models are loaded
    if MODEL_LOADING:
        return jsonify({
            "word": word,
            "neighbors": [],
            "error": "Models are still loading. Please try again in a moment."
        }), 503  # Service Unavailable

    if not MODELS_LOADED:
        return jsonify({
            "word": word,
            "neighbors": [],
            "error": "Models have not been loaded yet. Try again later."
        }), 503  # Service Unavailable

    neighbors = []
    error_message = None

    try:
        if embedding_type == 'glove':
            if GLOVE_MODEL and word.lower() in GLOVE_MODEL:
                # Get the embedding for the target word
                word_embedding = GLOVE_MODEL[word.lower()]
                
                # Create a list of words to compare against (all corpus words that are in GloVe)
                # For efficiency, we might want to limit this list in a production app
                comparison_words = [w for w in CORPUS_WORDS if w in GLOVE_MODEL]
                
                # Get embeddings for all comparison words
                comparison_embeddings = {w: GLOVE_MODEL[w] for w in comparison_words}
                
                # Find nearest neighbors
                neighbors = find_nearest_neighbors(
                    word_embedding, 
                    comparison_embeddings,
                    [word.lower()] + comparison_words,  # Include target word in the list
                    n_neighbors=n_neighbors
                )
            else:
                error_message = f"Word '{word}' not found in GloVe model."
        elif embedding_type == 'tfidf':
            if TFIDF_VECTORIZER:
                # Get TF-IDF embedding for the word
                word_embedding = get_word_tfidf_embedding(word.lower(), TFIDF_VECTORIZER, TFIDF_MATRIX)
                
                # If the vector is all zeros, the word isn't in our corpus
                if np.all(word_embedding == 0):
                    error_message = f"Word '{word}' not found in TF-IDF vocabulary or has no weight in the corpus."
                else:
                    # Get embeddings for all words in the vocabulary
                    comparison_words = TFIDF_VECTORIZER.get_feature_names_out().tolist()
                    comparison_embeddings = {}
                    
                    for w in comparison_words:
                        w_embedding = get_word_tfidf_embedding(w, TFIDF_VECTORIZER, TFIDF_MATRIX)
                        comparison_embeddings[w] = w_embedding
                    
                    # Find nearest neighbors
                    neighbors = find_nearest_neighbors(
                        word_embedding,
                        comparison_embeddings,
                        [word.lower()] + comparison_words,
                        n_neighbors=n_neighbors
                    )
            else:
                error_message = "TF-IDF model not loaded."
        else:
            error_message = f"Unsupported embedding type: {embedding_type}"

    except Exception as e:
        error_message = str(e)

    if error_message:
        return jsonify({
            "word": word,
            "neighbors": [],
            "error": error_message
        }), 404 if "not found" in error_message else 500

    return jsonify({
        "word": word,
        "neighbors": neighbors
    })

@app.route('/api/visualize_embeddings', methods=['GET'])
def get_visualized_embeddings():
    """
    Provides reduced dimension embeddings for visualization.
    Can be used to power t-SNE or PCA plots in the frontend.
    """
    method = request.args.get('method', 'tsne')  # 'tsne' or 'pca'
    embedding_type = request.args.get('type', 'glove')  # 'glove' or 'tfidf'
    
    # Check if models are loaded
    if MODEL_LOADING:
        return jsonify({
            "error": "Models are still loading. Please try again in a moment."
        }), 503  # Service Unavailable

    if not MODELS_LOADED:
        return jsonify({
            "error": "Models have not been loaded yet. Try again later."
        }), 503  # Service Unavailable
        
    reduced_data = []
    error_message = None
    
    try:
        if embedding_type == 'glove':
            if not GLOVE_MODEL:
                error_message = "GloVe model not loaded."
            else:
                # Select a subset of words from the corpus that exist in GloVe
                # For efficiency and visualization clarity, we'll limit to a reasonable number
                valid_words = []
                embedding_vectors = []
                
                # Get all corpus words that are in GloVe, up to a limit
                word_limit = 50  # Reasonable number for visualization
                for word in CORPUS_WORDS:
                    if word in GLOVE_MODEL and len(valid_words) < word_limit:
                        valid_words.append(word)
                        embedding_vectors.append(GLOVE_MODEL[word])
                
                if len(embedding_vectors) > 1:  # Need at least 2 points for dimensionality reduction
                    embedding_matrix = np.array(embedding_vectors)
                    reduced_matrix = reduce_dimensionality(embedding_matrix, method=method)
                    
                    # Format data for the frontend
                    reduced_data = [
                        {"word": valid_words[i], "x": float(reduced_matrix[i, 0]), "y": float(reduced_matrix[i, 1])}
                        for i in range(len(valid_words))
                    ]
                else:
                    error_message = "Not enough words with embeddings to visualize."
        
        elif embedding_type == 'tfidf':
            if not TFIDF_VECTORIZER or not TFIDF_MATRIX:
                error_message = "TF-IDF vectors not computed."
            else:
                # For TF-IDF, we'll visualize documents (sentences from our corpus)
                # Each document is represented by its TF-IDF vector
                
                # Convert sparse matrix to dense format
                dense_tfidf = TFIDF_MATRIX.toarray()
                
                if dense_tfidf.shape[0] > 1:  # Need at least 2 documents
                    # Apply dimensionality reduction
                    reduced_matrix = reduce_dimensionality(dense_tfidf, method=method)
                    
                    # Format data for the frontend, using a snippet of each document as the label
                    reduced_data = [
                        {
                            "word": custom_corpus[i][:30] + "..." if len(custom_corpus[i]) > 30 else custom_corpus[i],
                            "x": float(reduced_matrix[i, 0]),
                            "y": float(reduced_matrix[i, 1])
                        }
                        for i in range(dense_tfidf.shape[0])
                    ]
                else:
                    error_message = "Not enough documents for visualization."
        else:
            error_message = f"Unsupported embedding type: {embedding_type}"
                
    except Exception as e:
        error_message = str(e)
    
    if error_message:
        return jsonify({"error": error_message}), 500
    
    return jsonify({
        "embeddings_data": reduced_data,
        "method": method,
        "type": embedding_type
    })

@app.route('/api/status', methods=['GET'])
def get_model_status():
    """
    Endpoint to check the loading status of models.
    """
    return jsonify({
        "models_loaded": MODELS_LOADED,
        "loading_in_progress": MODEL_LOADING
    })

if __name__ == '__main__':
    # Start model loading in a background thread
    threading.Thread(target=load_models_background, daemon=True).start()
    
    # Running on a different port from assignment 1
    app.run(debug=True, port=5002) 