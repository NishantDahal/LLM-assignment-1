import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Dropout
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def build_encoder_decoder_model(input_vocab_size, output_vocab_size, 
                               input_max_len, output_max_len,
                               embedding_dim=128, lstm_units=256):
    """
    Build an encoder-decoder model with LSTM layers for text summarization
    
    Args:
        input_vocab_size: Size of the input vocabulary
        output_vocab_size: Size of the output vocabulary
        input_max_len: Maximum length of input sequences
        output_max_len: Maximum length of output sequences
        embedding_dim: Dimension of the embedding vectors
        lstm_units: Number of units in LSTM layers
        
    Returns:
        The compiled model
    """
    # Encoder
    encoder_inputs = Input(shape=(input_max_len,), name='encoder_inputs')
    encoder_embedding = Embedding(input_vocab_size, embedding_dim, 
                                  name='encoder_embedding')(encoder_inputs)
    
    # Using bidirectional LSTM for better context capture
    encoder_lstm = LSTM(lstm_units, return_state=True, name='encoder_lstm')
    _, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]
    
    # Decoder
    decoder_inputs = Input(shape=(output_max_len-1,), name='decoder_inputs')
    decoder_embedding = Embedding(output_vocab_size, embedding_dim, 
                                  name='decoder_embedding')(decoder_inputs)
    
    # Pass encoder states to initialize the decoder
    decoder_lstm = LSTM(lstm_units, return_sequences=True, return_state=True,
                        name='decoder_lstm')
    decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
    
    # Add a dropout layer to prevent overfitting
    decoder_dropout = Dropout(0.5, name='decoder_dropout')(decoder_outputs)
    
    # Dense output layer for word prediction
    decoder_dense = Dense(output_vocab_size, activation='softmax', 
                          name='decoder_dense')(decoder_dropout)
    
    # Define the model
    model = Model([encoder_inputs, decoder_inputs], decoder_dense)
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def create_inference_models(model, input_max_len, output_max_len, lstm_units):
    """
    Create encoder and decoder models for inference
    
    Args:
        model: The trained model
        input_max_len: Maximum length of input sequences
        output_max_len: Maximum length of output sequences
        lstm_units: Number of units in LSTM layers
        
    Returns:
        encoder_model, decoder_model
    """
    # Encoder model
    encoder_inputs = model.input[0]  # Input layer
    encoder_embedding = model.get_layer('encoder_embedding')(encoder_inputs)
    encoder_lstm = model.get_layer('encoder_lstm')
    _, state_h, state_c = encoder_lstm(encoder_embedding)
    encoder_states = [state_h, state_c]
    encoder_model = Model(encoder_inputs, encoder_states)
    
    # Decoder model
    decoder_inputs = model.input[1]  # Input layer
    decoder_state_input_h = Input(shape=(lstm_units,))
    decoder_state_input_c = Input(shape=(lstm_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    
    decoder_embedding = model.get_layer('decoder_embedding')(decoder_inputs)
    decoder_lstm = model.get_layer('decoder_lstm')
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_embedding, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    
    decoder_dropout = model.get_layer('decoder_dropout')(decoder_outputs)
    decoder_dense = model.get_layer('decoder_dense')
    decoder_outputs = decoder_dense(decoder_dropout)
    
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states
    )
    
    return encoder_model, decoder_model

def train_model(data_path, epochs=20, batch_size=64, validation_split=0.2):
    """
    Train the encoder-decoder model
    
    Args:
        data_path: Path to the preprocessed data
        epochs: Number of training epochs
        batch_size: Batch size for training
        validation_split: Portion of data to use for validation
        
    Returns:
        The trained model and history
    """
    # Load preprocessed data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    encoder_input = data['encoder_input']
    decoder_input = data['decoder_input']
    decoder_target = data['decoder_target']
    
    text_tokenizer = data['text_tokenizer']
    summary_tokenizer = data['summary_tokenizer']
    max_text_len = data['max_text_len']
    max_summary_len = data['max_summary_len']
    
    # Build model
    input_vocab_size = len(text_tokenizer.word_index) + 1
    output_vocab_size = len(summary_tokenizer.word_index) + 1
    
    model = build_encoder_decoder_model(
        input_vocab_size, 
        output_vocab_size,
        max_text_len,
        max_summary_len
    )
    
    # Print model summary
    model.summary()
    
    # Create callbacks for early stopping and model checkpointing
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5', save_best_only=True, monitor='val_loss')
    ]
    
    # Train the model
    history = model.fit(
        [encoder_input, decoder_input],
        np.expand_dims(decoder_target, -1),
        batch_size=batch_size,
        epochs=epochs,
        validation_split=validation_split,
        callbacks=callbacks
    )
    
    # Save model and tokenizers
    model.save('summarization_model.h5')
    
    model_data = {
        'text_tokenizer': text_tokenizer,
        'summary_tokenizer': summary_tokenizer,
        'max_text_len': max_text_len,
        'max_summary_len': max_summary_len
    }
    
    with open('model_data.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    # Create and save inference models
    encoder_model, decoder_model = create_inference_models(
        model, max_text_len, max_summary_len, lstm_units=256)
    
    encoder_model.save('encoder_model.h5')
    decoder_model.save('decoder_model.h5')
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    return model, history

if __name__ == "__main__":
    # Train the model
    model, history = train_model('preprocessed_data.pkl', epochs=10) 