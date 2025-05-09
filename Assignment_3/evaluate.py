import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import corpus_bleu
from rouge import Rouge
import pandas as pd

def decode_sequence(input_seq, encoder_model, decoder_model, 
                   max_summary_len, target_tokenizer):
    """
    Generate a summary for a given input sequence
    
    Args:
        input_seq: The input sequence to summarize
        encoder_model: The encoder model
        decoder_model: The decoder model
        max_summary_len: Maximum length of the summary
        target_tokenizer: Tokenizer for the summary vocabulary
        
    Returns:
        The generated summary
    """
    # Encode the input as state vectors
    states_value = encoder_model.predict(input_seq, verbose=0)
    
    # Generate empty target sequence of length 1 with start token
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_tokenizer.word_index.get('startseq', 1)
    
    # List to store generated tokens
    decoded_tokens = []
    
    # Generate summary one word at a time
    stop_condition = False
    while not stop_condition:
        # Predict next word
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value, verbose=0)
        
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, 0, :])
        sampled_word = ''
        
        # Get the word corresponding to the token index
        for word, index in target_tokenizer.word_index.items():
            if index == sampled_token_index:
                sampled_word = word
                break
        
        # Exit condition: either we hit max length or find end token
        if (sampled_word == 'endseq' or 
            len(decoded_tokens) > max_summary_len - 1):
            stop_condition = True
        else:
            decoded_tokens.append(sampled_word)
        
        # Update the target sequence (length 1)
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        
        # Update states
        states_value = [h, c]
    
    return ' '.join(decoded_tokens)

def generate_summaries(encoder_model, decoder_model, test_data, 
                      text_tokenizer, summary_tokenizer, max_summary_len,
                      num_samples=10):
    """
    Generate summaries for a set of test samples
    
    Args:
        encoder_model: The encoder model
        decoder_model: The decoder model
        test_data: The test data containing texts and summaries
        text_tokenizer: Tokenizer for the input text
        summary_tokenizer: Tokenizer for the summary
        max_summary_len: Maximum length of the summary
        num_samples: Number of samples to generate
        
    Returns:
        DataFrame with original text, reference summary and generated summary
    """
    # Select a subset of samples
    if len(test_data) > num_samples:
        indices = np.random.choice(len(test_data), num_samples, replace=False)
        test_subset = test_data.iloc[indices].reset_index(drop=True)
    else:
        test_subset = test_data
    
    results = []
    
    for i, row in test_subset.iterrows():
        text = row['clean_text']
        actual_summary = row['clean_summary'].replace('startseq ', '').replace(' endseq', '')
        
        # Tokenize and pad the input text
        seq = text_tokenizer.texts_to_sequences([text])
        padded_seq = pad_sequences(seq, maxlen=100, padding='post')
        
        # Generate summary
        generated_summary = decode_sequence(
            padded_seq, encoder_model, decoder_model, 
            max_summary_len, summary_tokenizer
        )
        
        # Print results
        print(f"Sample {i+1}:")
        print(f"Text: {text[:100]}...")
        print(f"Actual Summary: {actual_summary}")
        print(f"Generated Summary: {generated_summary}")
        print("-" * 50)
        
        results.append({
            'text': text,
            'actual_summary': actual_summary,
            'generated_summary': generated_summary
        })
    
    return pd.DataFrame(results)

def calculate_metrics(results_df):
    """
    Calculate BLEU and ROUGE scores for evaluation
    
    Args:
        results_df: DataFrame with actual and generated summaries
        
    Returns:
        Dictionary of metrics
    """
    # Prepare data for BLEU score
    references = []
    hypotheses = []
    
    for _, row in results_df.iterrows():
        actual = row['actual_summary'].split()
        generated = row['generated_summary'].split()
        
        references.append([actual])
        hypotheses.append(generated)
    
    # Calculate BLEU score
    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    
    # Calculate ROUGE score
    rouge = Rouge()
    
    # Format data for ROUGE
    actual_summaries = results_df['actual_summary'].tolist()
    generated_summaries = results_df['generated_summary'].tolist()
    
    # Handle empty summaries
    for i in range(len(generated_summaries)):
        if not generated_summaries[i]:
            generated_summaries[i] = "empty"
    
    try:
        rouge_scores = rouge.get_scores(generated_summaries, actual_summaries, avg=True)
    except:
        # If ROUGE fails (e.g., due to empty strings), return zero scores
        rouge_scores = {
            'rouge-1': {'f': 0.0, 'p': 0.0, 'r': 0.0},
            'rouge-2': {'f': 0.0, 'p': 0.0, 'r': 0.0},
            'rouge-l': {'f': 0.0, 'p': 0.0, 'r': 0.0}
        }
    
    # Compile metrics
    metrics = {
        'BLEU-1': bleu1,
        'BLEU-2': bleu2,
        'BLEU-3': bleu3,
        'BLEU-4': bleu4,
        'ROUGE-1-F': rouge_scores['rouge-1']['f'],
        'ROUGE-2-F': rouge_scores['rouge-2']['f'],
        'ROUGE-L-F': rouge_scores['rouge-l']['f']
    }
    
    return metrics

def evaluate_model(test_data_path, model_path, encoder_model_path, decoder_model_path):
    """
    Evaluate the model on test data
    
    Args:
        test_data_path: Path to the test data
        model_path: Path to the model data
        encoder_model_path: Path to the encoder model
        decoder_model_path: Path to the decoder model
    """
    # Load test data
    test_data = pd.read_pickle(test_data_path)
    
    # Load model data
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    
    text_tokenizer = model_data['text_tokenizer']
    summary_tokenizer = model_data['summary_tokenizer']
    max_summary_len = model_data['max_summary_len']
    
    # Load models
    encoder_model = load_model(encoder_model_path)
    decoder_model = load_model(decoder_model_path)
    
    # Generate summaries
    results_df = generate_summaries(
        encoder_model, decoder_model, test_data, 
        text_tokenizer, summary_tokenizer, max_summary_len
    )
    
    # Calculate metrics
    metrics = calculate_metrics(results_df)
    
    # Print metrics
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Save results
    results_df.to_csv('summary_results.csv', index=False)
    
    with open('evaluation_metrics.txt', 'w') as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    return results_df, metrics

if __name__ == "__main__":
    # Example usage
    evaluate_model(
        'test_data.pkl',
        'model_data.pkl',
        'encoder_model.h5',
        'decoder_model.h5'
    ) 