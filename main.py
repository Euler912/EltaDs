import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
#  Paths and configuration
# =========================================================

COMPETITION_NAME = "titanic"

DOWNLOAD_PATH = os.path.join("data", "raw")

TRAIN_FILE_NAME = "train.csv"

TRAIN_FILE_PATH = os.path.join(DOWNLOAD_PATH, TRAIN_FILE_NAME)

# Placeholder for the raw DataFrame
df_raw = None

# =========================================================
#  Downloading data
# =========================================================

# Create the download directory if it does not exist
if not os.path.exists(DOWNLOAD_PATH):
    os.makedirs(DOWNLOAD_PATH)
    print(f"Created directory: {DOWNLOAD_PATH}")

print(f"\n--- 1. Downloading and Extracting {COMPETITION_NAME} Dataset ---")

# Download the competition data using the Kaggle CLI
os.system(f"kaggle competitions download -c {COMPETITION_NAME} -p {DOWNLOAD_PATH}")

# Unzip the downloaded file into the same directory
os.system(f"unzip -o {DOWNLOAD_PATH}/{COMPETITION_NAME}.zip -d {DOWNLOAD_PATH}")

print("Download and extraction complete.")

# =========================================================
#  Loading data
# =========================================================

print("\n--- 2. Loading Data and Initial Check ---")

try:
    # Load the training CSV into a pandas DataFrame
    df_raw = pd.read_csv(TRAIN_FILE_PATH)

except FileNotFoundError:
    # In case the CSV is not found where expected
    print(f"\nFATAL ERROR: File not found at {TRAIN_FILE_PATH}.")
    print("Please ensure the Kaggle download and extraction were successful.")
except Exception as e:
    # Catch any other unexpected errors during loading
    print(f"\nAn unexpected error occurred during file loading: {e}")


# =========================================================
#  Feature engineering
# =========================================================

# Impute missing Age values with the median (robust to outliers)
age_median = df_raw['Age'].median()
df_raw['Age'] = df_raw['Age'].fillna(age_median)

# For Embarked (categorical), fill missing values with the most frequent port
most_frequent_port = df_raw['Embarked'].mode()[0]
df_raw['Embarked'] = df_raw['Embarked'].fillna(most_frequent_port)

# Cabin has many missing values, but still contains useful information (deck)
# Fill missing Cabin with 'Z' to represent "Unknown"
df_raw['Cabin'] = df_raw['Cabin'].fillna('Z')

# Extract the deck letter as a new feature (first character of Cabin)
df_raw['Deck'] = df_raw['Cabin'].str[0]

# Drop the original Cabin column to avoid sparsity and noise
df_raw.drop('Cabin', axis=1, inplace=True)
# Optionally, we could one-hot encode Embarked earlier:
# df_raw = pd.get_dummies(df_raw, columns=['Embarked'], prefix='Embarked')

# The Name column contains titles (Mr, Mrs, Miss, etc.) which are informative
# Extract title from the Name string using a regex
df_raw['Title'] = df_raw['Name'].str.extract(r'([A-Za-z]+)\.', expand=False)

# Map rare titles into a single "Rare" category and unify similar ones
title_mapping = {
    'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs',
    'Capt': 'Rare', 'Col': 'Rare', 'Major': 'Rare', 'Don': 'Rare',
    'Rev': 'Rare', 'Dr': 'Rare', 'Sir': 'Rare', 'Lady': 'Rare',
    'Countess': 'Rare', 'Jonkheer': 'Rare', 'Dona': 'Rare'
}
df_raw['Title'] = df_raw['Title'].replace(title_mapping)

# Create FamilySize = number of siblings/spouses + parents/children + self
df_raw['FamilySize'] = df_raw['SibSp'] + df_raw['Parch'] + 1

# Create a binary feature indicating whether the passenger is alone
df_raw['IsAlone'] = np.where(df_raw['FamilySize'] == 1, 1, 0)

# Drop columns that are IDs or high-cardinality text with low predictive power
df_raw.drop('Name', axis=1, inplace=True)        # Information now captured via Title
df_raw.drop('Ticket', axis=1, inplace=True)      # Irregular, high-cardinality strings
df_raw = df_raw.drop('PassengerId', axis=1)      # Pure identifier, no semantic meaning


# =========================================================
#  Encoding categorical variables
# =========================================================

# Map Sex to numeric values: male -> 0, female -> 1
sex_mapping = {'male': 0, 'female': 1}
df_raw['Sex'] = df_raw['Sex'].map(sex_mapping)

# These columns are treated as categorical and will be one-hot encoded
categorical_cols_to_encode = ['Pclass', 'Embarked', 'Deck', 'Title']

# One-hot encode the categorical columns
df_raw = pd.get_dummies(
    df_raw,
    columns=categorical_cols_to_encode,
    prefix=categorical_cols_to_encode,
    drop_first=False  # keep all categories to let the model learn its own baseline
)

# Convert any boolean columns (if present) to integers (0/1) for modeling
boolean_cols = df_raw.select_dtypes(include=['bool']).columns
df_raw[boolean_cols] = df_raw[boolean_cols].astype(int)


# =========================================================
#  Data visualization
# =========================================================

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# =========================================================
# 1. Class distribution of Survived
# =========================================================
axes[0].hist(df_raw['Survived'], bins=2, edgecolor='black')
axes[0].set_xticks([0, 1])
axes[0].set_xlabel("Survived")
axes[0].set_ylabel("Count")
axes[0].set_title("Distribution of Survival")

# =========================================================
# 2. Age distribution by Survival
# =========================================================
axes[1].hist(
    df_raw[df_raw['Survived'] == 0]['Age'],
    bins=20,
    alpha=0.5,
    label='Did not survive'
)
axes[1].hist(
    df_raw[df_raw['Survived'] == 1]['Age'],
    bins=20,
    alpha=0.5,
    label='Survived'
)
axes[1].set_xlabel('Age')
axes[1].set_ylabel('Count')
axes[1].set_title('Age Distribution by Survival')
axes[1].legend()

# =========================================================
# 3. Survival rate by Sex
# =========================================================
survival_by_sex = df_raw.groupby('Sex')['Survived'].mean()

# If Sex is encoded as 0/1, map to labels
sex_labels = ['Male' if s == 0 else 'Female' for s in survival_by_sex.index]

axes[2].bar(sex_labels, survival_by_sex.values)
axes[2].set_ylabel('Survival Rate')
axes[2].set_title('Survival Rate by Sex')

plt.tight_layout()
plt.show()

print(df_raw.head(3))