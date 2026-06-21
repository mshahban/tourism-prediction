from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError
import os

# Initialize connection layer using your specific custom token variable
api = HfApi(token=os.getenv("HF_TOKEN"))

# Define your unique Hugging Face Space repository parameters
repo_id = "mshahban/tourism-prediction-space"
repo_type = "space"

# Verify repository availability before running upload procedures
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Target Space '{repo_id}' already exists. Proceeding with sync...")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Provisions initiating...")
    # FIX: Changed space_sdk from "streamlit" to "docker" to correctly match your Dockerfile deployment path
    api.create_repo(repo_id=repo_id, repo_type=repo_type, space_sdk="docker", private=False)
    print(f"Space '{repo_id}' created successfully as a Docker space.")

print("Syncing deployment workspace files to Hugging Face...")
# Programmatically syncs your local deployment assets up to the cloud space
api.upload_folder(
    folder_path="tourism_prediction/deployment",
    repo_id=repo_id,
    repo_type=repo_type,
    path_in_repo="",
)
print(f"Deployment folder successfully synced to Hugging Face Space: https://huggingface.co{repo_id}")
