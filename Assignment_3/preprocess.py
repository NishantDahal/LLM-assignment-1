import pandas as pd
import numpy as np
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import pickle

def clean_text(text):
    """Clean and preprocess text"""
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        text = re.sub(r'\n', ' ', text)    # Remove newlines
        return text.strip()
    return ""

def load_and_preprocess_data(file_path, max_samples=200):
    """Load and preprocess the dataset"""
    try:
        # Try to read with utf-8 encoding
        df = pd.read_csv(file_path)
    except UnicodeDecodeError:
        # If utf-8 fails, try with latin-1 encoding
        df = pd.read_csv(file_path, encoding='latin-1')
    
    print(f"Dataset loaded with {len(df)} samples")
    print(f"Columns: {df.columns}")
    
    # Sample a subset of the data to make it manageable
    if len(df) > max_samples:
        df = df.sample(max_samples, random_state=42)
    
    # Identify text and summary columns
    # Assuming columns like 'text', 'content', or 'article' for text
    # and 'summary', 'headline', or 'title' for summary
    text_col = None
    summary_col = None
    
    possible_text_cols = ['text', 'content', 'article', 'news']
    possible_summary_cols = ['summary', 'headline', 'title']
    
    for col in df.columns:
        col_lower = col.lower()
        if text_col is None and any(text in col_lower for text in possible_text_cols):
            text_col = col
        if summary_col is None and any(summary in col_lower for summary in possible_summary_cols):
            summary_col = col
    
    if text_col is None or summary_col is None:
        print("Warning: Could not automatically identify text and summary columns")
        print(f"First few columns: {df.columns[:5]}")
        # Use first two columns as fallback
        if len(df.columns) >= 2:
            text_col = df.columns[0]
            summary_col = df.columns[1]
            print(f"Using {text_col} as text and {summary_col} as summary")
        else:
            raise ValueError("Dataset doesn't have enough columns")
    
    # Clean text and summaries
    df['clean_text'] = df[text_col].apply(clean_text)
    df['clean_summary'] = df[summary_col].apply(clean_text)
    
    # Add start and end tokens to summaries
    df['clean_summary'] = 'startseq ' + df['clean_summary'] + ' endseq'
    
    # Print statistics
    print(f"Number of samples: {len(df)}")
    print(f"Average text length (characters): {df['clean_text'].str.len().mean():.2f}")
    print(f"Average summary length (characters): {df['clean_summary'].str.len().mean():.2f}")
    
    return df[['clean_text', 'clean_summary']]

def tokenize_and_pad(texts, max_length, vocab_size=None):
    """Tokenize and pad a list of texts"""
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    
    return tokenizer, padded_sequences

def prepare_data_for_model(df, max_text_len=100, max_summary_len=15, vocab_size=10000):
    """Prepare data for the LSTM model"""
    # Tokenize texts and summaries
    text_tokenizer, padded_texts = tokenize_and_pad(
        df['clean_text'].values, max_text_len, vocab_size)
    
    summary_tokenizer, padded_summaries = tokenize_and_pad(
        df['clean_summary'].values, max_summary_len, vocab_size)
    
    print(f"Text vocabulary size: {len(text_tokenizer.word_index)}")
    print(f"Summary vocabulary size: {len(summary_tokenizer.word_index)}")
    
    # Create input and target data
    # Input: text sequences
    # Target: summary sequences
    encoder_input = padded_texts
    
    # Decoder inputs (shifted right by one position)
    decoder_input = padded_summaries[:, :-1]
    # Decoder targets
    decoder_target = padded_summaries[:, 1:]
    
    return {
        'encoder_input': encoder_input,
        'decoder_input': decoder_input,
        'decoder_target': decoder_target,
        'text_tokenizer': text_tokenizer,
        'summary_tokenizer': summary_tokenizer,
        'max_text_len': max_text_len,
        'max_summary_len': max_summary_len
    }

def save_preprocessed_data(data, output_path='preprocessed_data.pkl'):
    """Save preprocessed data to a file"""
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    file_path = "data/news_summary.csv"
    df = load_and_preprocess_data(file_path, max_samples=200)
    
    # Prepare data for model
    preprocessed_data = prepare_data_for_model(df)
    
    # Save preprocessed data
    save_preprocessed_data(preprocessed_data) 