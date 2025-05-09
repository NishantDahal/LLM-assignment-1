import nltk

# Download necessary NLTK data (run this once)
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, ne_chunk

# 1. Tokenization
def tokenize_text(text):
    """Tokenizes the input text."""
    tokens = word_tokenize(text)
    return tokens

# 2. Lemmatization
def lemmatize_tokens(tokens):
    """Lemmatizes a list of tokens."""
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmas

# 3. Stemming
def stem_tokens(tokens):
    """Stems a list of tokens."""
    stemmer = PorterStemmer()
    stems = [stemmer.stem(token) for token in tokens]
    return stems

# 4. POS Tagging
def pos_tag_tokens(tokens):
    """Performs Part-of-Speech tagging on a list of tokens."""
    tagged_tokens = pos_tag(tokens)
    return tagged_tokens

# 5. Named Entity Recognition (NER)
def named_entity_recognition(tagged_tokens):
    """Performs Named Entity Recognition on POS-tagged tokens."""
    # NER in NLTK works on POS-tagged sentences.
    # ne_chunk expects a list of (word, POS) tuples.
    tree = ne_chunk(tagged_tokens)
    return tree

# Comparison of Lemmatization and Stemming
def compare_stemming_lemmatization(words):
    """Compares stemming and lemmatization for a list of words."""
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    
    print(f"{'Word':<15} | {'Stem':<15} | {'Lemma':<15}")
    print("-" * 47)
    
    results = []
    for word in words:
        stem = stemmer.stem(word)
        lemma = lemmatizer.lemmatize(word, pos='v') # Using 'v' (verb) for better lemmatization in some cases
        results.append({'word': word, 'stem': stem, 'lemma': lemma})
        print(f"{word:<15} | {stem:<15} | {lemma:<15}")
    return results

if __name__ == '__main__':
    sample_text = "Apple is looking at buying U.K. startup for $1 billion. The quick brown foxes jumped over the lazy dogs."

    print("Original Text:", sample_text)
    
    # Tokenization
    tokens = tokenize_text(sample_text)
    print("Tokens:", tokens)
    
    # Lemmatization
    lemmas = lemmatize_tokens(tokens)
    print("Lemmas:", lemmas)
    
    # Stemming
    stems = stem_tokens(tokens)
    print("Stems:", stems)
    
    # POS Tagging
    # Note: For better POS tagging, it's often good to use the original casing.
    # However, for consistency with stemming/lemmatization, we'll use the tokenized (often lowercased) output.
    # For NER, original casing is generally preferred. Let's re-tokenize for POS and NER for better accuracy.
    raw_tokens_for_pos_ner = word_tokenize(sample_text)
    pos_tags = pos_tag_tokens(raw_tokens_for_pos_ner)
    print("POS Tags:", pos_tags)
    
    # Named Entity Recognition
    # NER typically works best with original casing and on POS-tagged sentences.
    ner_tree = named_entity_recognition(pos_tags)
    print("NER Tree:")
    # ner_tree.pretty_print() # This will print a tree structure. For API, a list of entities is better.
    
    # To extract entities from the tree:
    entities = []
    for subtree in ner_tree:
        if hasattr(subtree, 'label'):
            entity = " ".join([token for token, pos in subtree.leaves()])
            entities.append((entity, subtree.label()))
    print("Extracted Entities:", entities)

    print("--- Comparison of Stemming and Lemmatization ---")
    example_words = [
        "running", "ran", "runs",
        "better", "good",
        "corpora", "corpus",
        "mice", "mouse",
        "studies", "studying",
        "caring", "cares",
        "meeting", "meets", "met",
        "having", "had",
        "playing", "plays", "played",
        "programming", "programmer", "programs"
    ]
    comparison_results = compare_stemming_lemmatization(example_words)
    
    print("Explanation of Differences:")
    print("Stemming: Reduces words to their root form (stem) by chopping off suffixes. It's a crude heuristic process.")
    print("          Pros: Computationally faster. Simpler to implement.")
    print("          Cons: Can produce non-existent words (e.g., 'studi' from 'studies'). May group unrelated words or fail to group related ones.")
    print("          Example: 'running', 'ran', 'runs' might all become 'run' (or similar).")
    print("Lemmatization: Reduces words to their dictionary form (lemma) using vocabulary and morphological analysis.")
    print("          Pros: Produces actual words. More accurate representation of the word's base form.")
    print("          Cons: Computationally slower than stemming. Requires a dictionary/lexicon.")
    print("          Example: 'running' becomes 'run', 'ran' becomes 'run', 'better' becomes 'good' (if POS is considered).")
    print("Key Difference: Stemming is a rule-based, suffix-stripping process, while lemmatization is a more sophisticated, dictionary-based process that aims for the actual base form of a word.") 