"""Neural Network Code for the 3 less common diagnoses"""

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# load data
df = pd.read_csv('rare_df.csv')
df['TEXT'] = df['TEXT'].str.lower()

# split data
texts = df['TEXT']
labels = df['DIAGNOSIS']

# split into train, validation, and test 
texts_train, texts_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
texts_train, texts_validation, y_train, y_validation = train_test_split(texts_train, y_train, test_size=0.2, random_state=42)

# use TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.7, min_df=5, max_features=10000)
X_train = vectorizer.fit_transform(texts_train).toarray()
X_validation = vectorizer.transform(texts_validation).toarray()
X_test = vectorizer.transform(texts_test).toarray()

# idx to label
idx_to_label = {
    0: 'RENAL FAILURE',
    1: 'STROKE',
    2: 'WOUND INFECTION'
}

# labels to numeric value
label_to_idx = {label: idx for idx, label in idx_to_label.items()}  

y_train = y_train.map(label_to_idx).values
y_validation = y_validation.map(label_to_idx).values
y_test = y_test.map(label_to_idx).values

# define dataset for torch
class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = TextDataset(X_train, y_train)
validation_dataset = TextDataset(X_validation, y_validation)
test_dataset = TextDataset(X_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=64)
test_dataloader = DataLoader(test_dataset, batch_size=64)

# configure
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LSTM for Recurrent Neural Network
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=2, dropout=0.5):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        # add other layer
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # add dimension for the sequence
        x = x.unsqueeze(1)  
        _, (hidden, _) = self.lstm(x)
        # use the last hidden state
        hidden = hidden[-1]  
        hidden = self.relu(self.fc1(hidden))
        hidden = self.dropout(hidden)
        logits = self.fc2(hidden)
        return logits

input_size = X_train.shape[1]
hidden_size = 128
num_classes = len(idx_to_label) 

model = LSTMClassifier(input_size, hidden_size, num_classes).to(device)

# add weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# define loss and optimizer
loss_func = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train the function
def train_model(dataloader, model, loss_func, optimizer):
    model.train()
    total_loss = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = loss_func(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

# evaluate 
def evaluate_model(dataloader, model):
    model.eval()
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            predictions = outputs.argmax(1)
            correct += (predictions == y).sum().item()
            total += y.size(0)
            y_true.extend(y.cpu().numpy())  # true labels
            y_pred.extend(predictions.cpu().numpy())  # predicted labels
    return correct / total, y_true, y_pred

train_loss = []

# run through epochs
epochs = 20
for epoch in range(epochs):
    e_train_loss = train_model(train_dataloader, model, loss_func, optimizer)
    # append to loss
    train_loss.append(e_train_loss)
    val_accuracy, _, _ = evaluate_model(validation_dataloader, model)
    print(f"Epoch {epoch+1}/{epochs}, Avg, Loss: {e_train_loss:.4f}, Accuracy: {val_accuracy:.4f}")

# test model
test_accuracy, y_true, y_pred = evaluate_model(test_dataloader, model)
print(f"Test Accuracy: {test_accuracy:.4f}")

# classification report for top three diagnoses
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=[idx_to_label[i] for i in range(num_classes)]))


# use seaborn to plot
sns.set(style="whitegrid")

# make sure x and y lists are the same length
epochs_list = range(1, epochs + 1)
train_loss_list = train_loss if isinstance(train_loss, list) else [train_loss] * len(epochs_list)

# plot training loss
plt.figure(figsize=(8,6))
sns.lineplot(x=epochs_list, y=train_loss_list, label = 'Training Loss', marker = 'o', color = 'orange' )
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Avg. Loss', fontsize=14)
plt.title(f"Training Loss Over {epochs} Epochs For 3 Less Common Diagnoses ", fontsize=16)
plt.xticks(epochs_list)
plt.legend(fontsize=10)
plt.grid(True, linestyle='-', alpha=0.7)

# save plot
plt.savefig('RNN_training_loss_plot_lesscommon.png')
plt.show()
