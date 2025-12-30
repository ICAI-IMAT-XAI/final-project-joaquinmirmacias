import os

# --- CREDENTIALS ---
# User and API key de Kaggle
os.environ['KAGGLE_USERNAME'] = ""
os.environ['KAGGLE_KEY'] = ""

# ---  IMPORT Y AUTENTICATION ---
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# --- DOWNLOAD ---
dataset = 'hassnainzaidi/ai-art-vs-human-art'
path_destino = '.'

print(f"Starting {dataset}...")

# unzip=True unzip after the download
api.dataset_download_files(dataset, path=path_destino, unzip=True)

print("¡Download!")