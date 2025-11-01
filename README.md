# text_classification_finetune_bert

# Directory_structure

project_root/
├── data/
│   ├── column_descriptions_v1.csv
│   ├── sensitivity_labels_v3.csv
│   ├── unseen_test_data.csv
│   └── NBX_Bank_Data_Security_guidelines_v1.pdf
├── fine_tuned_roberta_weighted/  <- Will be created by the training script
│   ├── pytorch_model.bin
│   └── ... (tokenizer and config files)
├── train_model.py               <- Script 1 (for training, conceptually)
└── predict_data.py              <- Script 2 (for prediction)



# Training and Fine-Tuning the Model
The first script (which we'll call train_model.py) handles data preparation, RoBERTa fine-tuning, evaluation, and saving the model.

# Key Features of the Training Script:
Input Format: Combines column name and description: "{column_name} [SEP] {column_description}".

Weighted Loss: 
Calculates and uses balanced class weights to address the imbalance between sensitive (1) and non-sensitive (0) columns.

Model Saving:
 Saves the final fine-tuned model and tokenizer to the ./fine_tuned_roberta_weighted directory.

# Prediction and Evaluation on Unseen Data
The second script (which we'll call predict_data.py) loads the saved model and evaluates its performance on the unseen_test_data.csv file.

Usage

Ensure the fine_tuned_roberta_weighted directory exists (i.e., you have run the training script successfully).

Ensure unseen_test_data.csv is in the ./data directory.

Execute the prediction script:

Bash

python predict_data.py
Output
The script will print:

The loaded unseen_test_df DataFrame.

Overall evaluation metrics on the unseen data (Accuracy, Precision, Recall, F1 Score).

A detailed Classification Report for classes 0 (Not Sensitive) and 1 (Sensitive).

A list of predictions showing the Column Name, Description, and Predicted Sensitivity.