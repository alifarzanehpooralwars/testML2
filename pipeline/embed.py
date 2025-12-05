"""
Embedding pipeline for semantic search service:
    1- Loads data (ids and abstracts)
    2- Uses a model from sentence transformer to calculate the embeddings
    3- Saves the embeddings + IDs + metadata to separate files
    4- Logs run information to mlflow
"""

import datetime
import time

import json
import os

import numpy as np

import mlflow

from sentence_transformers import SentenceTransformer

from .config import (
    ABSTRACTS_FILE,
    MODEL_NAME,
    MODELS_DIR,
    IDS_FILE,
    EMBEDDINGS_FILE,
    METADATA_FILE,
)

from .data_loader import  load_abstracts

# A simple flag to disable mlflow for Docker builds.
DISABLE_MLFLOW = os.environ.get("DISABLE_MLFLOW", "0") == "1"

def generate_embeddings() -> None:
    """
    - Loads abstracts
    - Calculats embeddings with sentence transformer
    - Saves embeddings, IDS, and metadata to disk
    - Logs to mlflow
    """

    df = load_abstracts()

    ids = df["id"].tolist()
    abstracts = df["abstract"].tolist()

    print(f"Loaded {len(ids)} abstracts from {ABSTRACTS_FILE}")

    print(f"Loading model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    def _embed_all():
        print(f"Embedding {len(ids)} abstracts from {ABSTRACTS_FILE}")
        start = time.time()
        embeddings = model.encode(abstracts, show_progress_bar=True)
        duration = time.time() - start
        print(f"Embedding took {duration} seconds")
        return embeddings, duration

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if DISABLE_MLFLOW:
        print("MLFlow disabled (DISABLE_MLFLOW=1)")
        embeddings, duration = _embed_all()
    else:
        # Set up mlflow experiment
        mlflow.set_experiment("semantic_search_embeddings")

        with mlflow.start_run(run_name="semantic_search_embeddings"):
            embeddings, duration = _embed_all()

            print(f"Saving embeddings to {EMBEDDINGS_FILE}")
            np.save(EMBEDDINGS_FILE, embeddings)

            print(f"Saving IDs to {IDS_FILE}")
            ids_array = np.array(ids, dtype=int)
            np.save(IDS_FILE, ids_array)

            metadata = {
                "model_name": MODEL_NAME,
                "num_documents": len(ids),
                "embedding_dimension": int(embeddings.shape[1]),
                "created_at": datetime.datetime.utcnow().isoformat(),
            }

            print(f"Saving metadata to {METADATA_FILE}")
            with open(METADATA_FILE, "w") as f:
                json.dump(metadata, f, indent=2)

            # Log parameters, metrics and artifacts to mlflow
            mlflow.log_param("model_name", MODEL_NAME)
            mlflow.log_param("num_documents", len(ids))
            mlflow.log_param("embedding_dimension", int(embeddings.shape[1]))
            mlflow.log_param("embedding_duration_seconds", duration)

            mlflow.log_artifact(str(EMBEDDINGS_FILE))
            mlflow.log_artifact(str(IDS_FILE))
            mlflow.log_artifact(str(METADATA_FILE))
            print("MLFlow logging complete")
            print("Finished")
            return

    print("Saving embeddings to {EMBEDDINGS_FILE}")
    np.save(EMBEDDINGS_FILE, embeddings)

    print(f"Saving IDs to {IDS_FILE}")
    ids_array = np.array(ids, dtype=int)
    np.save(IDS_FILE, ids_array)

    metadata = {
        "model_name": MODEL_NAME,
        "num_documents": len(abstracts),
        "embedding_dimension": int(embeddings.shape[1]),
        "created_at": datetime.datetime.now().isoformat(),
    }

    print(f"Saving metadata to {METADATA_FILE}")
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)

    print("Finished!")


if __name__ == "__main__":
    generate_embeddings()





