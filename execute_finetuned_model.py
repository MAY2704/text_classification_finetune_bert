import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
import os
import numpy as np 

# Define local directories for model and data
LOCAL_MODEL_DIR = './fine_tuned_roberta_weighted'
LOCAL_DATA_DIR = './data'

# Ensure the local data directory exists for the test file (if not already done)
if not os.path.exists(LOCAL_DATA_DIR):
    os.makedirs(LOCAL_DATA_DIR, exist_ok=True)
    print(f"Created local data directory: {LOCAL_DATA_DIR}")

def predict_unseen_data_sensitivity(model_path, unseen_data_path):
    """
    Predicts sensitivity of unseen data using a fine-tuned RoBERTa model.

    Args:
        model_path (str): Path to the directory containing the fine-tuned RoBERTa model.
        unseen_data_path (str): Path to the CSV file containing the unseen data.

    Returns:
        None: Prints the evaluation metrics (accuracy, precision, recall, F1 score)
              and the classification report, as well as predictions.
    """

    # Load the fine-tuned RoBERTa model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        loaded_roberta_model_weighted = RobertaForSequenceClassification.from_pretrained(model_path).to(device)
        loaded_roberta_tokenizer_weighted = RobertaTokenizer.from_pretrained(model_path)
    except Exception as e:
        print(f"Error: Failed to load model or tokenizer from {model_path}. Check the path and ensure the files are present. Error details: {e}")
        return

    # Function to prepare input for RoBERTa
    def prepare_bert_input(column_name, column_description, tokenizer, max_length=128):
        text = f"{column_name} [SEP] {column_description}"
        inputs = tokenizer(text, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
        return inputs['input_ids'], inputs['attention_mask']

    # Load unseen test data
    try:
        unseen_test_df = pd.read_csv(unseen_data_path, sep=';')
    except FileNotFoundError:
        print(f"Error: Unseen data.csv not found at {unseen_data_path}. Please make sure the file exists and the path is correct.")
        return
    except Exception as e:
        print(f"Error: An error occurred while reading the unseen test data: {e}")
        return
    print(unseen_test_df) #keep the print

    # Prepare data for prediction
    if 'column_name' not in unseen_test_df.columns or 'description' not in unseen_test_df.columns or 'is_sensitive' not in unseen_test_df.columns:
        print("Error: The unseen test data file is missing required columns ('column_name', 'description', or 'is_sensitive').")
        return

    unseen_column_names = unseen_test_df['column_name'].tolist()
    unseen_column_descriptions = unseen_test_df['description'].fillna('').tolist()
    unseen_labels = unseen_test_df['is_sensitive'].map({'Y': 1, 'N': 0}).tolist()
    unseen_labels_tensor = torch.tensor(unseen_labels, dtype=torch.long)

    # Prepare inputs for the unseen data
    unseen_input_ids = []
    unseen_attention_masks = []
    for name, desc in zip(unseen_column_names, unseen_column_descriptions):
        ids, masks = prepare_bert_input(name, desc, loaded_roberta_tokenizer_weighted)
        unseen_input_ids.append(ids)
        unseen_attention_masks.append(masks)

    unseen_input_ids = torch.cat(unseen_input_ids, dim=0)
    unseen_attention_masks = torch.cat(unseen_attention_masks, dim=0)

    # Create DataLoader for unseen test data
    batch_size = 32
    unseen_test_dataset = TensorDataset(unseen_input_ids, unseen_attention_masks, unseen_labels_tensor)
    unseen_test_dataloader = DataLoader(unseen_test_dataset, sampler=SequentialSampler(unseen_test_dataset), batch_size=batch_size)

    # Make predictions on the unseen test data
    loaded_roberta_model_weighted.eval()
    unseen_predictions = []
    unseen_true_labels = []

    for batch in unseen_test_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        with torch.no_grad():
            outputs = loaded_roberta_model_weighted(b_input_ids, attention_mask=b_input_mask)
        logits = outputs.logits
        _, predicted = torch.max(logits, dim=1)
        unseen_predictions.extend(predicted.cpu().numpy())
        unseen_true_labels.extend(b_labels.cpu().numpy())

    # Calculate metrics

    try:
        unseen_precision, unseen_recall, unseen_f1, _ = precision_recall_fscore_support(unseen_true_labels, unseen_predictions, average='binary', zero_division=0)
    except ValueError:
        # Fallback to 'weighted' if there's an issue with 'binary' (e.g., only one class present)
        unseen_precision, unseen_recall, unseen_f1, _ = precision_recall_fscore_support(unseen_true_labels, unseen_predictions, average='weighted', zero_division=0)
        print("Note: Fell back to 'weighted' average due to possible single-class data in unseen set.")


    print("\n--- Results on Unseen Test Data ---")
    print(f"Accuracy: {accuracy_score(unseen_true_labels, unseen_predictions):.4f}")
    print(f"Precision: {unseen_precision:.4f}, Recall: {unseen_recall:.4f}, F1 Score: {unseen_f1:.4f}")
    print(classification_report(unseen_true_labels, unseen_predictions, zero_division=0))

    # Print the predictions along with column names and descriptions
    print("\n--- Predictions ---")
    for i, prediction in enumerate(unseen_predictions):
        sensitivity_label = "Sensitive (Y)" if prediction == 1 else "Not Sensitive (N)"
        print(f"Column Name: {unseen_column_names[i]}, Description: {unseen_column_descriptions[i]}, Predicted Sensitivity: {sensitivity_label}")


if __name__ == "__main__":
    # Example usage using local paths:
    # IMPORTANT: Ensure your fine-tuned model files (pytorch_model.bin, config.json, vocab.json, etc.)
    # are in the LOCAL_MODEL_DIR, and your CSV file is in the LOCAL_DATA_DIR.
    model_path = LOCAL_MODEL_DIR
    unseen_data_filename = 'unseen_test_data.csv' # Assuming this file exists in LOCAL_DATA_DIR
    unseen_data_path = os.path.join(LOCAL_DATA_DIR, unseen_data_filename)

    # Check if the paths exist
    if not os.path.exists(model_path):
        print(f"Error: Model path '{model_path}' does not exist. Please run the model training script first to save the model.")
    elif not os.path.exists(unseen_data_path):
        print(f"Error: Unseen data path '{unseen_data_path}' does not exist. Please place the '{unseen_data_filename}' file inside the '{LOCAL_DATA_DIR}' folder.")
    else:
        print("Loading model and predicting sensitivity for unseen data...")
        predict_unseen_data_sensitivity(model_path, unseen_data_path)