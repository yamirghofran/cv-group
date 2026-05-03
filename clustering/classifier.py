"""Team classifier: SigLIP image embeddings → UMAP(3D) → KMeans(k=2).

Vendored from `roboflow/sports@feat/basketball` (`sports/common/team.py`,
Apache-2.0). Vendored rather than installed so we can later swap the embedder,
reducer, and clusterer for ablation experiments without depending on an
unreleased upstream branch.

Pipeline (matches the Roboflow basketball-ai notebook):
    crops → SigLIP vision features → UMAP(n_components=3) → KMeans(n_clusters=2)
"""

from __future__ import annotations

from collections.abc import Generator, Iterable
from typing import TypeVar

import numpy as np
import supervision as sv
import torch
import umap
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import AutoProcessor, SiglipVisionModel

V = TypeVar("V")

SIGLIP_MODEL_PATH = "google/siglip-base-patch16-224"


def create_batches(
    sequence: Iterable[V], batch_size: int
) -> Generator[list[V], None, None]:
    """Yield successive `batch_size` chunks from `sequence`."""
    batch_size = max(batch_size, 1)
    current_batch: list[V] = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch


class TeamClassifier:
    """Cluster player crops into two teams using SigLIP + UMAP + KMeans."""

    def __init__(
        self,
        device: str = "cpu",
        batch_size: int = 32,
        n_teams: int = 2,
        random_state: int = 42,
    ) -> None:
        if n_teams < 2:
            raise ValueError("n_teams must be >= 2")
        self.device = device
        self.batch_size = batch_size
        self.n_teams = n_teams
        self.random_state = random_state
        self.features_model = SiglipVisionModel.from_pretrained(
            SIGLIP_MODEL_PATH
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
        self.reducer = umap.UMAP(n_components=3, random_state=random_state)
        self.cluster_model = KMeans(
            n_clusters=n_teams, n_init="auto", random_state=random_state
        )

    def extract_features(self, crops: list[np.ndarray]) -> np.ndarray:
        """Embed crops to a (N, D) feature matrix using mean-pooled SigLIP features."""
        crops = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = create_batches(crops, self.batch_size)
        data = []
        with torch.no_grad():
            for batch in tqdm(batches, desc="Embedding extraction"):
                inputs = self.processor(images=batch, return_tensors="pt").to(
                    self.device
                )
                outputs = self.features_model(**inputs)
                embeddings = (
                    torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                )
                data.append(embeddings)

        return np.concatenate(data)

    def fit(self, crops: list[np.ndarray]) -> None:
        """Fit the classifier on a list of player crops."""
        data = self.extract_features(crops)
        projections = self.reducer.fit_transform(data)
        self.cluster_model.fit(projections)

    def predict(self, crops: list[np.ndarray]) -> np.ndarray:
        """Predict team labels (0 or 1) for a list of player crops."""
        if len(crops) == 0:
            return np.array([])

        data = self.extract_features(crops)
        projections = self.reducer.transform(data)
        return self.cluster_model.predict(projections)

    def fit_predict(self, crops: list[np.ndarray]) -> np.ndarray:
        """Fit on `crops` and return their team labels in one feature-extraction pass.

        Equivalent in result to `fit(crops)` followed by `predict(crops)`, but
        runs SigLIP only once. Use this when you want labels for the same set
        you fit on (the common single-clip CLI case). For separate train and
        inference sets, call `fit` and `predict` independently.
        """
        if len(crops) == 0:
            return np.array([])

        data = self.extract_features(crops)
        projections = self.reducer.fit_transform(data)
        return self.cluster_model.fit_predict(projections)
