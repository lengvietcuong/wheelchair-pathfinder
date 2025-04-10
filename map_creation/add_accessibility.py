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


MAP_DIRECTORY = Path("map")
SLOPES_PATH = MAP_DIRECTORY / "adjacency_matrix_slope.csv"
KERB_RAMPS_PATH = MAP_DIRECTORY / "adjacency_matrix_kerb_ramps.csv"
SIDEWALK_WIDTH_PATH = MAP_DIRECTORY / "adjacency_matrix_sidewalk_width.csv"
NODE_FEATURES_PATH = MAP_DIRECTORY / "adjacency_matrix_node_features.csv"


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

    # Load adjacency matrix
    df = pd.read_csv(ADJACENCY_MATRIX_PATH, index_col=0).replace("inf", np.inf)
    locations = df.index.tolist()

    # Create feature matrices
    slope_df = pd.DataFrame(index=locations, columns=locations)
    kerb_ramps_df = pd.DataFrame(index=locations, columns=locations)
    sidewalk_width_df = pd.DataFrame(index=locations, columns=locations)
    for i in locations:
        for j in locations:
            distance = df.loc[i, j]
            if np.isinf(distance) or distance <= 0:  # Skip if no path
                continue
            
            # The slope angle is either 0° (50% chance) or a random angle between 0° and 45°
            slope_df.loc[i, j] = (
                0 if random.random() < 0.5 else round(random.uniform(0, 45), 1)
            )
            # Kerp ramps are either present (1) or absent (0), with 50/50 chance
            kerb_ramps_df.loc[i, j] = random.randint(0, 1)
            # Sidewalk width randomized between 0.9m and 3.0m
            sidewalk_width_df.loc[i, j] = round(random.uniform(0.9, 3.0), 1)

    # Create accessibility features for each location
    # Each feature has a 25% chance of being present
    node_features = {
        loc: {
            "has_accessible_restroom": random.random() < 0.25,
            "has_accessible_parking": random.random() < 0.25,
            "has_accessible_entrance": random.random() < 0.25,
            "has_rest_area": random.random() < 0.25,
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
