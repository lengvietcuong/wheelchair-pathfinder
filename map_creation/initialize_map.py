"""
Map Initialization Helper.
"""

import json
import logging

import pandas as pd
import numpy as np

from map_creation.add_accessibility import (
    generate_accessibility_features,
    KERB_RAMPS_PATH,
    NODE_FEATURES_PATH,
    SIDEWALK_WIDTH_PATH,
    SLOPES_PATH,
)
from map_creation.map_to_matrix import (
    ADJACENCY_MATRIX_PATH,
    NODES_COORDINATES_PATH,
    extract_kmz_file,
)


logger = logging.getLogger(__name__)


def initialize_map():
    """Initialize the map by loading the adjacency matrix, node coordinates, and accessibility features.

    Returns:
        dict: A dictionary containing:
            - adjacency_matrix: The adjacency matrix of the map.
            - node_coordinates: The coordinates of the nodes.
            - slope_matrix: The slope matrix of the map.
            - kerb_ramps_matrix: The kerb ramps matrix of the map.
            - sidewalk_width_matrix: The sidewalk width matrix of the map.
            - node_features: The node features matrix of the map.
    """

    # Create adjacency matrix and node coordinates files if needed
    if not ADJACENCY_MATRIX_PATH.exists() or not NODES_COORDINATES_PATH.exists():
        logger.info("Adjacency matrix and/or node coordinates not found. Creating...")
        extract_kmz_file()

    # Add accessibility features if needed
    if not all(
        file.exists()
        for file in (
            SLOPES_PATH,
            KERB_RAMPS_PATH,
            SIDEWALK_WIDTH_PATH,
            NODE_FEATURES_PATH,
        )
    ):
        logger.info("Accessibility feature matrices not found. Creating...")
        generate_accessibility_features()

    # Load the adjacency matrix and node coordinates
    adjacency_matrix = pd.read_csv(ADJACENCY_MATRIX_PATH, index_col=0)
    adjacency_matrix = adjacency_matrix.replace("inf", np.inf)
    with open(NODES_COORDINATES_PATH) as file:
        node_coordinates = json.load(file)

    # Load the accessibility feature matrices
    slope_matrix = pd.read_csv(SLOPES_PATH, index_col=0)
    kerb_ramps_matrix = pd.read_csv(KERB_RAMPS_PATH, index_col=0)
    sidewalk_width_matrix = pd.read_csv(SIDEWALK_WIDTH_PATH, index_col=0)
    node_features = pd.read_csv(NODE_FEATURES_PATH, index_col=0)

    return {
        "adjacency_matrix": adjacency_matrix,
        "node_coordinates": node_coordinates,
        "slope_matrix": slope_matrix,
        "kerb_ramps_matrix": kerb_ramps_matrix,
        "sidewalk_width_matrix": sidewalk_width_matrix,
        "node_features": node_features,
    }
