import os
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle

# Import our modules
from preprocess import load_and_preprocess_data, prepare_data_for_model, save_preprocessed_data
from model import train_model
from evaluate import evaluate_model

def run_summarization_pipeline(data_path, test_size=0.2, max_samples=200, epochs=10, batch_size=32):
    """
    Run the complete summarization pipeline from preprocessing to evaluation
    
    Args:
        data_path: Path to the news summary CSV dataset
        test_size: Proportion of data to use for testing
        max_samples: Maximum number of samples to use from the dataset
        epochs: Number of training epochs
        batch_size: Batch size for training
    """
    print("=" * 50)
    print("ABSTRACTIVE TEXT SUMMARIZATION PIPELINE")
    print("=" * 50)
    
    # Step 1: Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    df = load_and_preprocess_data(data_path, max_samples=max_samples)
    
    # Step 2: Split data into train and test sets
    print("\n2. Splitting data into train and test sets...")
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    print(f"Training samples: {len(train_df)}")
    print(f"Testing samples: {len(test_df)}")
    
    # Step 3: Prepare data for the model
    print("\n3. Preparing data for the model...")
    preprocessed_data = prepare_data_for_model(train_df)
    
    # Save preprocessed data
    print("Saving preprocessed data...")
    save_preprocessed_data(preprocessed_data, 'preprocessed_data.pkl')
    
    # Save test data for later evaluation
    print("Saving test data...")
    test_df.to_pickle('test_data.pkl')
    
    # Step 4: Train the model
    print("\n4. Training the model...")
    model, history = train_model('preprocessed_data.pkl', epochs=epochs, batch_size=batch_size)
    
    # Step 5: Evaluate the model
    print("\n5. Evaluating the model...")
    try:
        results_df, metrics = evaluate_model(
            'test_data.pkl',
            'model_data.pkl',
            'encoder_model.h5',
            'decoder_model.h5'
        )
        
        # Print summary of results
        print("\nModel Evaluation Summary:")
        print(f"BLEU-1 Score: {metrics['BLEU-1']:.4f}")
        print(f"ROUGE-L F-Score: {metrics['ROUGE-L-F']:.4f}")
        print(f"Examples: {len(results_df)} summaries generated")
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
    
    print("\nPipeline completed successfully!")
    print("Check the following files for results:")
    print("- summary_results.csv: Generated summaries")
    print("- evaluation_metrics.txt: Evaluation metrics")
    print("- training_history.png: Training loss and accuracy curves")

if __name__ == "__main__":
    # Set the file path
    data_path = "data/news_summary.csv"
    
    # Ensure the file exists
    if not os.path.exists(data_path):
        print(f"Error: File not found at {data_path}")
        print("Please make sure the dataset is in the correct location.")
        exit(1)
    
    # Check if required packages are installed
    try:
        import tensorflow as tf
        import nltk
        from rouge import Rouge
        
        # Download required NLTK data
        nltk.download('punkt')
        
    except ImportError as e:
        print(f"Error: Missing required packages - {str(e)}")
        print("Please install the required packages using:")
        print("pip install tensorflow nltk pandas rouge scikit-learn matplotlib")
        exit(1)
    
    # Run the pipeline
    run_summarization_pipeline(
        data_path=data_path,
        max_samples=200,  # Use 200 samples as specified in the task
        epochs=10,        # Reduced epochs for faster completion
        batch_size=32     # Adjust based on available memory
    ) 