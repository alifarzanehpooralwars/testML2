from pathlib import Path



BASE_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = BASE_DIR / "data"

ABSTRACTS_FILE = DATA_DIR / "arxiv_abstracts.jsonl"

MODEL_NAME = "sentence-transformers/all-MiniLM-L12-v2"

MODELS_DIR = BASE_DIR / "models"
EMBEDDINGS_FILE = MODELS_DIR / "embeddings.npy"
IDS_FILE = MODELS_DIR / "ids.npy"
METADATA_FILE = MODELS_DIR / "metadata.json"

