# Titanic Survival – PyTorch & Streamlit

This project trains a simple feed-forward neural network to predict passenger survival on the Titanic.  

The workflow is split into two main parts:

1. **EDA (Main.py) ** – performs exploratory data analysis and feature engineering and returns a **non-normalized dataset**.  
   - The final cleaned DataFrame is called **`raw_df`** (in the training script it is read as `df_raw` from CSV).
2. **Training (Train.py) ** – uses a **Tkinter file dialog** to let you manually select the preprocessed CSV (`raw_df`), then:
   - creates train / validation / test splits,
   - normalizes selected numeric features,
   - trains the neural network,
   - plots metrics (loss, confusion matrix, ROC),

---

## Dependencies

The project uses the following Python packages:

```python
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
