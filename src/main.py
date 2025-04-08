import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

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
from path_finding.a_star import AccessibilitySetting, AStar


logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    data_folder = Path("data")

    # Create adjacency matrix and node coordinates files if needed
    if not ADJACENCY_MATRIX_PATH.exists() or not NODES_COORDINATES_PATH.exists():
        logger.info("Adjacency matrix or node coordinates not found. Creating them...")
        try:
            extract_kmz_file()
        except Exception as error:
            logger.error(f"Error extracting KMZ file: {error}", exc_info=True)
            return

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
        logger.info("Accessibility feature matrices not found. Creating them...")
        try:
            generate_accessibility_features()
        except Exception as error:
            logger.error(f"Error generating accessibility features: {error}")

    # Load the adjacency matrix and node coordinates
    try:
        adjacency_matrix = pd.read_csv(
            data_folder / "adjacency_matrix.csv", index_col=0
        )
        adjacency_matrix = adjacency_matrix.replace("inf", np.inf)
    except Exception as error:
        logger.error(f"Failed to load adjacency matrix: {error}", exc_info=True)
        return
    try:
        with open(data_folder / "node_coordinates.json", "r") as file:
            node_coordinates = json.load(file)
    except Exception as error:
        logger.error(f"Failed to load node coordinates: {error}", exc_info=True)
        return

    # Load the accessibility feature matrices
    try:
        slope_matrix = pd.read_csv(
            data_folder / "adjacency_matrix_slope.csv", index_col=0
        )
        kerb_ramps_matrix = pd.read_csv(
            data_folder / "adjacency_matrix_kerb_ramps.csv", index_col=0
        )
        sidewalk_width_matrix = pd.read_csv(
            data_folder / "adjacency_matrix_sidewalk_width.csv", index_col=0
        )
    except Exception as error:
        logger.error(
            f"Error loading accessibility feature matrices: {error}", exc_info=True
        )
        slope_matrix = None
        kerb_ramps_matrix = None
        sidewalk_width_matrix = None

    # Find a path
    start = "Noi Due Cafe"
    goal = "Robert Lehman Collection Library"
    pathfinder = AStar(
        adjacency_matrix=adjacency_matrix,
        node_coordinates=node_coordinates,
        slope_matrix=slope_matrix,
        kerb_ramps_matrix=kerb_ramps_matrix,
        sidewalk_width_matrix=sidewalk_width_matrix,
    )

    logger.info(
        f"Finding path from '{start}' to '{goal}' "
        "with wheelchair accessibility considered..."
    )
    result = pathfinder.find_path(
        start, goal, accessibility=AccessibilitySetting.COST_AND_HEURISTIC
    )
    logger.info(f"Path: {result.path}")
    logger.info(f"Cost: {result.cost:.2f}")
    logger.info(f"Number of nodes created: {result.nodes_created}")


if __name__ == "__main__":
    main()
