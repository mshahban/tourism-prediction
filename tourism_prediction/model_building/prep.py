# Standard packages for data handling
import pandas as pd
import os

# Scikit-learn utilities for splitting and encoding data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Hugging Face interface client
from huggingface_hub import HfApi

# Setup Hugging Face connection layer
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/mshahban/tourism-prediction-dataset/tourism.csv"

# Load the source dataset from Hugging Face Hub
df = pd.read_csv(DATASET_PATH)
print("Data asset successfully retrieved from the Hub.")

# Remove arbitrary index columns if present in the source file
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

# Remove the primary key column since it holds no predictive power
df.drop(columns=['CustomerID'], inplace=True)

# Identify and purge identical records from the dataframe
dupe_records = df.duplicated().sum()
print(f"Removing {dupe_records} redundant rows from the workspace...")
df = df.drop_duplicates()

# Missing values auditing block
missing_entries = df.isna().sum().sum()
print(f"Total remaining missing data fields: {missing_entries}")

# Standardize inconsistent categorical string variations
cleaning_maps = {
    'Gender': {'Fe Male': 'Female'},
    'MaritalStatus': {'Unmarried': 'Single'}
}
df.replace(cleaning_maps, inplace=True)

# Segment continuous Age values into uniform 5-year binned intervals
age_intervals = [18, 24, 29, 34, 39, 44, 49, 54, 59, 65]
interval_names = ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-65']
df['AgeGroup'] = pd.cut(df['Age'], bins=age_intervals, labels=interval_names, include_lowest=True).astype(str)
df.drop(columns=['Age'], inplace=True)

# Isolate feature types based on unique element counts
cat_features = df.select_dtypes(include=['object', 'category']).columns
two_value_features = [col for col in cat_features if df[col].nunique() == 2]
multi_value_features = [col for col in cat_features if df[col].nunique() > 2]

# Map binary categories into machine-readable numeric formats (0 or 1)
encoder_engine = LabelEncoder()
for col in two_value_features:
    df[col] = encoder_engine.fit_transform(df[col])

# Expand multi-class nominal categories using dummy flags
df = pd.get_dummies(df, columns=multi_value_features, dtype=int)

# Define the target classification label
target_label = 'ProdTaken'

# Segment datasets into independent features and target masks
X = df.drop(columns=[target_label])
y = df[target_label]

# Run stratified evaluation partitioning split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Export processed frames to temporary local disk storage
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# Re-upload split artifacts securely back to Hugging Face
split_files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]
for file_name in split_files:
    api.upload_file(
        path_or_fileobj=file_name,
        path_in_repo=file_name.split("/")[-1],  # just the filename,
        repo_id="mshahban/tourism-prediction-dataset",
        repo_type="dataset",
    )
print("Data Preparation phase successfully synced to Hugging Face space!")
