import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



# Load your dataset from the Excel file
df = pd.read_excel('Clearways_train_data_small.xlsx', sheet_name='Sheet1')
input_data = df.iloc[:, 1].apply(str)
ground_truth = df.iloc[:, 0].apply(str)

# Split the data into train and test (90% train, 10% test)
torch.manual_seed(42)
train_size = int(0.9 * len(input_data))
test_size = len(input_data) - train_size
input_train_dataset, input_test_dataset = random_split(input_data, [train_size, test_size])
gt_train_dataset, gt_test_dataset = random_split(ground_truth, [train_size, test_size])


access_token = "a"
# Load the Llama 3 tokenizer
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
tokenizer.add_special_tokens({"pad_token":"<pad>"})


# Tokenize input data
encoded_input_train = tokenizer(list(input_train_dataset), padding=True, truncation=True, return_tensors='pt', max_length=1024)
encoded_gt_train = tokenizer(list(gt_train_dataset), padding=True, truncation=True, return_tensors='pt', max_length=1024)
encoded_input_test = tokenizer(list(input_test_dataset), padding=True, truncation=True, return_tensors='pt', max_length=1024)
encoded_gt_test = tokenizer(list(gt_test_dataset), padding=True, truncation=True, return_tensors='pt', max_length=1024)

train_dataset = TensorDataset(encoded_input_train['input_ids'], encoded_input_train['attention_mask'], encoded_gt_train['input_ids'])
test_dataset = TensorDataset(encoded_input_test['input_ids'], encoded_input_test['attention_mask'], encoded_gt_test['input_ids'])


# Set the batch size
batch_size = 528

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


model = AutoModelForCausalLM.from_pretrained(model_name, num_labels=2, token=access_token)  # Adjust num_labels as needed
model.config.pad_token_id = model.config.eos_token_id


# Set up optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)
criterion = nn.CrossEntropyLoss()

best_loss = float('inf')
patience = 3
no_improvement = 0
model.train()
# Training loop
for epoch in range(25):  # Adjust the number of epochs as needed
    print(f"Epoch {epoch + 1}")
    total_loss = 0
    for batch in train_loader:
        input_ids, attention_mask, labels = batch
        
        print(f"Batch size: {len(input_ids)}")
        print(f"Labels shape: {labels.shape}")
        print(f"Attention Mask size: {len(attention_mask)}")

        optimizer.zero_grad()

        print(input_ids.shape)
        print(attention_mask.shape)
        print(labels.shape)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        print(f"Outputs shape: {outputs.logits.shape}")
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Calculate average loss for the epoch
    avg_loss = total_loss / len(train_loader)

    if avg_loss < best_loss:
            best_loss = avg_loss
            no_improvement = 0
            # Save the model checkpoint
            model.save_pretrained('./fine_tuned_model')
            tokenizer.save_pretrained('./fine_tuned_model')
    else:
        no_improvement += 1
        if no_improvement >= patience:
            print(f"Early stopping: Loss has not improved for {patience} epochs.")
            break


    # Print epoch information
        print(f"Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}, Best Loss: {best_loss:.4f}")

# Print final results
print(f"Training completed. Best loss achieved: {best_loss:.4f}")

# def test_model(model, test_loader):
#     model.eval()
#     all_preds = []
#     all_labels = []

#     with torch.no_grad():
#         for batch in test_loader:
#             input_ids, attention_mask, labels = batch
#             outputs = model(input_ids, attention_mask=attention_mask)
#             logits = outputs.logits
#             preds = torch.argmax(logits, dim=1)
#             all_preds.extend(preds.tolist())
#             all_labels.extend(labels.tolist())

#     accuracy = accuracy_score(all_labels, all_preds)
#     precision = precision_score(all_labels, all_preds, average='weighted')
#     recall = recall_score(all_labels, all_preds, average='weighted')
#     f1 = f1_score(all_labels, all_preds, average='weighted')

#     return accuracy, precision, recall, f1

# # Test the model using the test_loader
# accuracy, precision, recall, f1 = test_model(model, test_loader)

# print(f"Accuracy: {accuracy:.4f}")
# print(f"Precision: {precision:.4f}")
# print(f"Recall: {recall:.4f}")
# print(f"F1 Score: {f1:.4f}")