from flask import Flask, request, jsonify, render_template
# Make sure nlp_utils is in the same directory or Python path
from nlp_utils import (
    tokenize_text,
    lemmatize_tokens,
    stem_tokens,
    pos_tag_tokens,
    named_entity_recognition,
    compare_stemming_lemmatization
)
import nltk

# It's good practice to ensure NLTK data is available.
# These lines can be run once, or kept here to ensure availability if the environment changes.
# Consider moving to a setup script or Dockerfile for production.
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("NLTK 'punkt' not found. Downloading...")
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("NLTK 'wordnet' not found. Downloading...")
    nltk.download('wordnet')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    print("NLTK 'averaged_perceptron_tagger' not found. Downloading...")
    nltk.download('averaged_perceptron_tagger')
try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    print("NLTK 'maxent_ne_chunker' not found. Downloading...")
    nltk.download('maxent_ne_chunker')
try:
    nltk.data.find('corpora/words')
except LookupError:
    print("NLTK 'words' not found. Downloading...")
    nltk.download('words')


app = Flask(__name__)

@app.route('/')
def index():
    # This will serve the demo web app
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_text():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing "text" field in request'}), 400
    
    text_input = data['text']
    operation = data.get('operation', 'all') # Default to all operations

    response = {'original_text': text_input}

    # Always tokenize first, as most other operations depend on tokens
    # For POS and NER, it's better to tokenize the original text to preserve case
    tokens_for_pos_ner = tokenize_text(text_input) 
    # For stemming/lemmatization, we often lowercase, but tokenize_text doesn't do that by default.
    # NLTK's stemmers/lemmatizers can handle mixed case, but sometimes lowercasing first is preferred.
    # For this API, we'll use the direct tokens.
    tokens = tokenize_text(text_input)


    if operation == 'tokenize' or operation == 'all':
        response['tokens'] = tokens
    
    if operation == 'lemmatize' or operation == 'all':
        response['lemmas'] = lemmatize_tokens(tokens)
        
    if operation == 'stem' or operation == 'all':
        response['stems'] = stem_tokens(tokens)

    if operation == 'pos_tag' or operation == 'all':
        # POS tagging works best on tokens that preserve original casing
        pos_tags = pos_tag_tokens(tokens_for_pos_ner)
        response['pos_tags'] = pos_tags

    if operation == 'ner' or operation == 'all':
        # NER also requires POS-tagged tokens, preferably with original casing
        pos_tags_for_ner = pos_tag_tokens(tokens_for_pos_ner)
        ner_tree = named_entity_recognition(pos_tags_for_ner)
        
        entities = []
        for subtree in ner_tree:
            if hasattr(subtree, 'label'):
                entity = " ".join([token for token, pos in subtree.leaves()])
                entities.append((entity, subtree.label()))
            # Include tokens that are not part of an entity as well
            # else:
            #     if isinstance(subtree, tuple): # It's a (token, POS) pair
            #         entities.append((subtree[0], "O")) # "O" for Outside an entity
        response['ner'] = entities

    return jsonify(response)

@app.route('/compare_stem_lemma', methods=['POST'])
def compare_stem_lemma_api():
    data = request.get_json()
    if not data or 'words' not in data or not isinstance(data['words'], list):
        return jsonify({'error': 'Missing "words" field (must be a list) in request'}), 400

    words_to_compare = data['words']
    if len(words_to_compare) == 0 :
        return jsonify({'error': '"words" list cannot be empty'}), 400
    
    # The compare_stemming_lemmatization function in nlp_utils prints to console.
    # For an API, we want to return the data.
    # We'll adapt it slightly or call its core logic.
    
    stemmer = nltk.stem.PorterStemmer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    
    results = []
    for word in words_to_compare:
        stem = stemmer.stem(word)
        # Attempt to lemmatize as a verb first, then as a noun if it doesn't change
        lemma_v = lemmatizer.lemmatize(word.lower(), pos='v')
        lemma_n = lemmatizer.lemmatize(word.lower(), pos='n')
        # Choose the more "changed" version, or verb version if both are same as word
        lemma = lemma_v if lemma_v != word.lower() else lemma_n 
        if lemma == word.lower() and word.endswith('s'): # simple plural check
             lemma = lemmatizer.lemmatize(word.lower())


        results.append({'word': word, 'stem': stem, 'lemma': lemma})
        
    explanation = [
        "Stemming: Reduces words to their root form (stem) by chopping off suffixes. It's a crude heuristic process.",
        "Pros: Computationally faster. Simpler to implement.",
        "Cons: Can produce non-existent words. May group unrelated words or fail to group related ones.",
        "Lemmatization: Reduces words to their dictionary form (lemma) using vocabulary and morphological analysis.",
        "Pros: Produces actual words. More accurate representation of the word's base form.",
        "Cons: Computationally slower. Requires a dictionary/lexicon.",
        "Key Difference: Stemming is rule-based suffix-stripping; lemmatization is dictionary-based aiming for the actual base form."
    ]

    return jsonify({'comparison': results, 'explanation': explanation})

if __name__ == '__main__':
    app.run(debug=True, port=5001) # Using port 5001 to avoid potential conflicts 