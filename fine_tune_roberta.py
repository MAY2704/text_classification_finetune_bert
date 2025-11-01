!pip install transformers
!pip install nltk
!pip install PyPDF2

import pandas as pd
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup, RobertaTokenizer, RobertaForSequenceClassification
import torch.optim as optim
AdamW = optim.AdamW
import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
import PyPDF2

import nltk
try:
    nltk.data.find('tokenizers/punkt')
except getattr(nltk, 'exceptions', Exception):
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except getattr(nltk, 'exceptions', Exception):
    nltk.download('wordnet')
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except getattr(nltk, 'exceptions', Exception):
    nltk.download('averaged_perceptron_tagger')

# Define local directories
DATA_DIRECTORY = './data'
OUTPUT_DIR = './fine_tuned_roberta_sensitivity'
OUTPUT_DIR_WEIGHTED = './fine_tuned_roberta_weighted'

# Create output directories if they don't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
if not os.path.exists(OUTPUT_DIR_WEIGHTED):
    os.makedirs(OUTPUT_DIR_WEIGHTED)

# Load data files from the local DATA_DIRECTORY
try:
    column_info_df = pd.read_csv(os.path.join(DATA_DIRECTORY, 'column_descriptions_v1.csv'), sep=';')
    sensitivity_labels_df = pd.read_csv(os.path.join(DATA_DIRECTORY, 'sensitivity_labels_v3.csv'), sep=';')
except FileNotFoundError as e:
    print(f"Error loading files. Please ensure the 'data' directory exists and contains the required files: {e}")
    exit()

print(column_info_df)
print(sensitivity_labels_df)

# Prepare Data for Fine-tuning RoBERTa
merged_df = pd.merge(column_info_df, sensitivity_labels_df, on='column_name', how='inner')
if 'is_sensitive' not in merged_df.columns:
    print("Error: 'is_sensitive' column not found in sensitivity labels file.")
    exit()

merged_df['label'] = merged_df['is_sensitive'].map({'Y': 1, 'N': 0})
merged_df['label'] = merged_df['label'].astype(int)

column_names = merged_df['column_name'].tolist()
column_descriptions = merged_df['description'].fillna('').tolist()
labels = merged_df['label'].tolist()

# Load Pre-trained RoBERTa Model and Tokenizer for Sequence Classification
roberta_model_name = 'roberta-base'
roberta_tokenizer = RobertaTokenizer.from_pretrained(roberta_model_name)
roberta_model = RobertaForSequenceClassification.from_pretrained(roberta_model_name, num_labels=len(np.unique(labels)))

# Prepare Input for RoBERTa Fine-tuning
def prepare_bert_input(column_name, column_description, tokenizer, max_length=128):
    text = f"{column_name} [SEP] {column_description}"
    inputs = tokenizer(text, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
    return inputs['input_ids'], inputs['attention_mask']

input_ids = []
attention_masks = []
for name, desc in zip(column_names, column_descriptions):
    ids, masks = prepare_bert_input(name, desc, roberta_tokenizer)
    input_ids.append(ids)
    attention_masks.append(masks)

input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels_tensor = torch.tensor(labels, dtype=torch.long)

# Split Data for Fine-tuning
train_inputs, temp_inputs, train_labels, temp_labels = train_test_split(
    input_ids, labels_tensor, test_size=0.3, random_state=42
)
train_masks, temp_masks, _, _ = train_test_split(
    attention_masks, labels_tensor, test_size=0.3, random_state=42
)
val_inputs, test_inputs, val_labels, test_labels = train_test_split(
    temp_inputs, temp_labels, test_size=0.5, random_state=42
)
val_masks, test_masks, _, _ = train_test_split(
    temp_masks, temp_labels, test_size=0.5, random_state=42
)

# Create DataLoaders
batch_size = 32

train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
train_dataloader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=batch_size)

val_dataset = TensorDataset(val_inputs, val_masks, val_labels)
val_dataloader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=batch_size)

test_dataset = TensorDataset(test_inputs, test_masks, test_labels)
test_dataloader = DataLoader(test_dataset, sampler=SequentialSampler(test_dataset), batch_size=batch_size)

# Fine-tuning the RoBERTa Model with Weighted Loss
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
roberta_model.to(device)

optimizer = AdamW(roberta_model.parameters(), lr=2e-5)
epochs = 8
total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Calculate Class Weights
train_labels_np = train_labels.cpu().numpy()
class_weights = compute_class_weight('balanced', classes=np.unique(train_labels_np), y=train_labels_np)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
print("Class Weights:", class_weights_tensor)

print("\n--- Fine-tuning RoBERTa Model with Weighted Loss ---")
for epoch in range(epochs):
    roberta_model.train()
    total_loss = 0
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        roberta_model.zero_grad()
        outputs = roberta_model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        logits = outputs.logits

        # Calculate loss with class weights
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
        loss = loss_fn(logits, b_labels)

        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(roberta_model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} Training Loss: {avg_train_loss:.4f}")

    # Evaluation on Validation Set
    roberta_model.eval()
    val_preds = []
    val_true = []
    val_loss = 0
    for batch in val_dataloader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        with torch.no_grad():
            outputs = roberta_model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            val_loss += loss.item()
            logits = outputs.logits
            _, predicted = torch.max(logits, dim=1)
            val_preds.extend(predicted.cpu().numpy())
            val_true.extend(b_labels.cpu().numpy())

    avg_val_loss = val_loss / len(val_dataloader)
    val_accuracy = accuracy_score(val_true, val_preds)
    print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Evaluation on Test Set
roberta_model.eval()
test_preds = []
test_true = []
test_loss = 0
for batch in test_dataloader:
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_labels = batch[2].to(device)
    with torch.no_grad():
        outputs = roberta_model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
        loss = outputs.loss
        test_loss += loss.item()
        logits = outputs.logits
        _, predicted = torch.max(logits, dim=1)
        test_preds.extend(predicted.cpu().numpy())
        test_true.extend(b_labels.cpu().numpy())

avg_test_loss = test_loss / len(test_dataloader)
test_accuracy = accuracy_score(test_true, test_preds)

# Calculate precision, recall, and F1-score
precision, recall, f1, _ = precision_recall_fscore_support(test_true, test_preds, average='binary')

print(f"\n--- RoBERTa Model Test Results with Weighted Loss ---")
print(f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
print("\nTest Classification Report:")
print(classification_report(test_true, test_preds))

# Save the Fine-tuned Model
roberta_model.save_pretrained(OUTPUT_DIR_WEIGHTED)
roberta_tokenizer.save_pretrained(OUTPUT_DIR_WEIGHTED)
print(f"\nFine-tuned RoBERTa model with weighted loss saved to: {OUTPUT_DIR_WEIGHTED}")

# Function to extract keywords from PDF using PyPDF2
def extract_keywords_from_pdf(pdf_path):
    keywords = []
    try:
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                if text:
                    # Simple keyword extraction (can be improved)
                    words = text.split()
                    keywords.extend(words)  # Add all words as keywords
    except FileNotFoundError:
        print(f"Error: PDF file not found at {pdf_path}")
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return keywords

# The PDF file is now expected to be in the local DATA_DIRECTORY
pdf_file_path = os.path.join(DATA_DIRECTORY, 'a.pdf')
extracted_keywords = extract_keywords_from_pdf(pdf_file_path)
if extracted_keywords:
    print("\nExtracted Keywords from PDF:")
    print(extracted_keywords)
else:
    print("\nNo keywords extracted from PDF.")


# Prompting
def predict_sensitivity_roberta_weighted(column_name, column_description, tokenizer, model, device, max_length=128):
    inputs = tokenizer(f"{column_name} [SEP] {column_description}", max_length=max_length, padding='max_length', truncation=True, return_tensors='pt').to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    return predicted_class, confidence

# Load the saved weighted RoBERTa model and tokenizer for inference
loaded_roberta_model_weighted = RobertaForSequenceClassification.from_pretrained(OUTPUT_DIR_WEIGHTED).to(device)
loaded_roberta_tokenizer_weighted = RobertaTokenizer.from_pretrained(OUTPUT_DIR_WEIGHTED)