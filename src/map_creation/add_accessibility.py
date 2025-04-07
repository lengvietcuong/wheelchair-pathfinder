"""
Random Accessibility Features Generator for Wheelchair Navigation.
"""

import logging
import random
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from .map_to_matrix import ADJACENCY_MATRIX_PATH


DATA_DIRECTORY = Path("data")
SLOPES_PATH = DATA_DIRECTORY / "adjacency_matrix_slope.csv"
KERB_RAMPS_PATH = DATA_DIRECTORY / "adjacency_matrix_kerb_ramps.csv"
SIDEWALK_WIDTH_PATH = DATA_DIRECTORY / "adjacency_matrix_sidewalk_width.csv"
NODE_FEATURES_PATH = DATA_DIRECTORY / "adjacency_matrix_node_features.csv"


logger = logging.getLogger(__name__)


def generate_accessibility_features(
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Enhance a map's adjacency matrix with random accessibility features and save to CSV files.

    Parameters:
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        Dict[str, pd.DataFrame]: Original and modified matrices plus node features.
    """
    # Set random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Load adjacency matrix and replace 'inf'
    df = pd.read_csv(ADJACENCY_MATRIX_PATH, index_col=0).replace("inf", np.inf)
    locations = df.index.tolist()

    # Initialize feature matrices
    slope_df = pd.DataFrame(index=locations, columns=locations)
    kerb_ramps_df = pd.DataFrame(index=locations, columns=locations)
    sidewalk_width_df = pd.DataFrame(index=locations, columns=locations)

    # Populate feature matrices
    for i in locations:
        for j in locations:
            dist = df.loc[i, j]
            if dist == np.inf or dist <= 0:
                continue
            slope_df.loc[i, j] = round(random.uniform(0, 15), 1)
            kerb_ramps_df.loc[i, j] = random.randint(0, 1)
            sidewalk_width_df.loc[i, j] = round(random.uniform(0.9, 2.5), 1)

    # Generate node-level accessibility features
    node_features = {
        loc: {
            "has_accessible_restroom": random.choice([True, False]),
            "has_accessible_parking": random.choice([True, False]),
            "has_accessible_entrance": random.choice([True, False]),
            "has_rest_area": random.choice([True, False]),
        }
        for loc in locations
    }
    node_features_df = pd.DataFrame.from_dict(node_features, orient="index")

    # Compile results
    results: Dict[Path, pd.DataFrame] = {
        SLOPES_PATH: slope_df,
        KERB_RAMPS_PATH: kerb_ramps_df,
        SIDEWALK_WIDTH_PATH: sidewalk_width_df,
        NODE_FEATURES_PATH: node_features_df,
    }

    # Save each DataFrame to CSV
    for file_path, table in results.items():
        table.to_csv(file_path)
        logging.info("Saved %s", file_path)
    return results
