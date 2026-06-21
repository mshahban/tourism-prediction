from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi
import os

# Define your Hugging Face configuration
repo_id = "mshahban/tourism-prediction-dataset"
repo_type = "dataset"
token = os.getenv("HF_TOKEN")

# Initialize the Hugging Face API client with token
api = HfApi(token=token)

# Step 1: Check if the repository exists, if not create it using the api object
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    # FIX: Use api.create_repo instead of the standalone create_repo function
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created successfully.")

# Step 2: Upload the data folder
print("Starting folder upload...")
api.upload_folder(
    folder_path="tourism_prediction/data",
    repo_id=repo_id,
    repo_type=repo_type,
)
print("Upload completed successfully!")
