import torch
import streamlit as st
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
)
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import random
import tkinter as tk
from tkinter import filedialog




# =========================================================
# 1. SEEDING — FULL DETERMINISM
# =========================================================
# Fix seeds to make results as reproducible as possible
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# =========================================================
# 2. READ CSV
# =========================================================
# Use a simple Tkinter file dialog so the user can select
# the preprocessed Titanic CSV file interactively.
root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename(
    title="Select Titanic CSV File",
    filetypes=[("CSV Files", "*.csv")]
)

if not file_path:
    raise ValueError("No file selected!")

print("File selected:", file_path)

# Load the processed CSV created in the previous script
print("The dataset so far (before normalization) is given by: "
      )

df_raw = pd.read_csv(file_path)
print(df_raw.head())

# =========================================================
# 3. Train / Val / Test split
# =========================================================
# Separate features (X) and target (y)
X = df_raw.drop('Survived', axis=1)
y = df_raw['Survived'].values  # 1D array for stratify

# Split into temp + test (80% / 20%), preserving class proportions
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Split temp further into train + validation
# 0.25 of 80% = 20%, so final: 60% train / 20% val / 20% test
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp
)




# =========================================================
# 4. Scaling (fit on train only, apply to val/test)
# =========================================================
# Only scale continuous numeric columns. Categorical one-hot
# columns remain in {0,1} and should NOT be scaled.
SCALING_COLS = ['Age', 'Fare', 'FamilySize']

scaler = StandardScaler()
X_train[SCALING_COLS] = scaler.fit_transform(X_train[SCALING_COLS])
X_val[SCALING_COLS]   = scaler.transform(X_val[SCALING_COLS])
X_test[SCALING_COLS]  = scaler.transform(X_test[SCALING_COLS])

# =========================================================
# 5. Convert NumPy arrays to Torch tensors
# =========================================================
X_train_np = X_train.to_numpy().astype(np.float32)
X_val_np   = X_val.to_numpy().astype(np.float32)
X_test_np  = X_test.to_numpy().astype(np.float32)

# Targets are binary (0/1), shaped as column vectors
y_train_np = y_train.astype(np.float32).reshape(-1, 1)
y_val_np   = y_val.astype(np.float32).reshape(-1, 1)
y_test_np  = y_test.astype(np.float32).reshape(-1, 1)

X_train_t = torch.from_numpy(X_train_np)
X_val_t   = torch.from_numpy(X_val_np)
X_test_t  = torch.from_numpy(X_test_np)

y_train_t = torch.from_numpy(y_train_np)
y_val_t   = torch.from_numpy(y_val_np)
y_test_t  = torch.from_numpy(y_test_np)




# =========================================================
# 6. Model definition (simple feedforward NN)
# =========================================================
class Net(nn.Module):
    """
    Simple fully-connected neural network for binary classification
    on the Titanic dataset.

    Architecture:
    - Input layer: size = number of features
    - Hidden layer: Linear -> ReLU -> Dropout
    - Output layer: single neuron + Sigmoid for probability of survival
    """
    def __init__(self, input_size, hidden_size, dropout_rate):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.dropout_rate = dropout_rate

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        # Dropout applied only during training; helps reduce overfitting
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.fc2(x)
        # Sigmoid to output a probability in [0, 1]
        x = torch.sigmoid(x)
        return x

# =========================================================
# 7. Training function
# =========================================================
def train_and_track(net, train_loader, val_loader, criterion, optimizer, num_epochs):
    """
    Train the model and track training/validation loss per epoch.

    """
    train_loss_history = []
    val_loss_history = []

    print(f"--- Training for {num_epochs} epochs ---")

    for epoch in range(num_epochs):
        net.train()
        epoch_train_loss = 0.0

        # --------- Training loop ---------
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item() * inputs.size(0)

        train_loss_history.append(epoch_train_loss / len(train_loader.dataset))

        # --------- Validation loop ---------
        net.eval()
        with torch.no_grad():
            epoch_val_loss = 0.0
            for inputs, labels in val_loader:
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item() * inputs.size(0)
            val_loss_history.append(epoch_val_loss / len(val_loader.dataset))

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] | "
                f"Train Loss: {train_loss_history[-1]:.4f} | "
                f"Val Loss: {val_loss_history[-1]:.4f}"
            )

    return train_loss_history, val_loss_history

# =========================================================
# 8. Evaluation helpers: Loss curves, Confusion Matrix, ROC
# =========================================================
def plot_loss_history(train_loss, val_loss):
    """Plot training vs validation loss across epochs."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (BCE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_confusion_matrix_and_metrics(net, X_tensor, y_tensor):
    """
    Compute predictions on the given set, plot the confusion matrix,
    and print overall accuracy.
    """
    net.eval()
    with torch.no_grad():
        outputs = net(X_tensor)
        # Threshold at 0.5 to convert probabilities to class labels
        y_pred_tensor = (outputs > 0.5).int()

    y_true = y_tensor.numpy().flatten().astype(int)
    y_pred = y_pred_tensor.numpy().flatten().astype(int)

    cm = confusion_matrix(y_true, y_pred)
    labels = ['Died (0)', 'Survived (1)']

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Confusion Matrix (Test Set)')
    plt.show()

    accuracy = np.mean(y_true == y_pred)
    print(f'\n>>> Test Accuracy: {accuracy * 100:.2f}%')


def plot_roc_curve(net, X_tensor, y_tensor):
    """
    Compute and plot ROC curve and AUC on the given set.
    """
    net.eval()
    with torch.no_grad():
        y_scores = net(X_tensor).numpy().flatten()
        y_true = y_tensor.numpy().flatten()

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    print(f"ROC AUC on Test Set: {roc_auc:.4f}")

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test Set)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()

# =========================================================
# 9. Main run block
# =========================================================
if __name__ == "__main__":

    try:
        input_size = X_train_t.shape[1]
    except NameError:
        print("\n Error: tensors not defined. Check data loading section.")
        exit()

    # -------------------------
    # Hyperparameters
    # -------------------------
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10   # Small number of epochs to reduce overfitting on this small dataset
    HIDDEN_SIZE = 128
    DROPOUT_RATE = 0.1

    # IMPORTANT!!!

    # - Adam is used because it is a very common and effective optimizer in DL
    #   for training neural networks, especially on smaller tabular datasets.
    # - The learning rate, hidden size, and dropout rate were chosen
    #   via manual trial-and-error to balance convergence speed and performance.
    # - Using only 10 epochs helps limit overfitting given the small
    #   size of the Titanic dataset.

    # DataLoaders for mini-batch training
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Instantiate model, loss function, and optimizer
    model = Net(input_size, HIDDEN_SIZE, DROPOUT_RATE)

    # Binary Cross-Entropy is appropriate for binary classification
    criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # -------------------------
    # Train the model
    # -------------------------
    train_loss, val_loss = train_and_track(
        model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS
    )

    # -------------------------
    # Evaluation on test set
    # -------------------------
    plot_loss_history(train_loss, val_loss)
    plot_confusion_matrix_and_metrics(model, X_test_t, y_test_t)
    plot_roc_curve(model, X_test_t, y_test_t)

    # -------------------------
    # Save trained model weights
    # -------------------------
    MODEL_SAVE_PATH = 'models/titanic_model.pth'
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
