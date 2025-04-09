import json
import logging
from typing import Dict

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
from path_finding.a_star import AStar
from path_finding.benchmark import run_full_benchmark
from path_finding.bfs import BFS
from path_finding.custom_types import Algorithm
from path_finding.dfs import DFS
from path_finding.path_finder import PathFinder


PATH_FINDERS: Dict[Algorithm, PathFinder] = {
    Algorithm.A_STAR: AStar,
    Algorithm.DFS: DFS,
    Algorithm.BFS: BFS,
}
TEST_CASE_COUNT = 20
RUNS_PER_TEST_CASE = 10


logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
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

    return {
        "adjacency_matrix": adjacency_matrix,
        "node_coordinates": node_coordinates,
        "slope_matrix": slope_matrix,
        "kerb_ramps_matrix": kerb_ramps_matrix,
        "sidewalk_width_matrix": sidewalk_width_matrix,
    }


def main():
    def handle_run_path_finder():
        """Take user input for start and goal locations, run the path finder, and display the outputs."""

        print("Available algorithms:")
        for index, algorithm in enumerate(PATH_FINDERS, start=1):
            print(f"{index}: {algorithm.value}")

        algorithm_index = int(input("\n>>> Your choice: ").strip()) - 1
        path_finders = list(PATH_FINDERS.values())
        path_finder = path_finders[algorithm_index](
            adjacency_matrix=adjacency_matrix,
            node_coordinates=node_coordinates,
            slope_matrix=slope_matrix,
            kerb_ramps_matrix=kerb_ramps_matrix,
            sidewalk_width_matrix=sidewalk_width_matrix,
        )

        start = input("Enter start location (e.g. Noi Due Cafe): ").strip()
        goal = input(
            "Enter goal location (e.g. Robert Lehman Collection Library): "
        ).strip()
        consider_accessibility = (
            input("Consider wheelchair accessibility? (y/n): ").strip().lower() == "y"
        )

        result = path_finder.find_path(
            start, goal, consider_accessibility=consider_accessibility
        )
        print(f"\nPath: {" → ".join(result.path)}")
        print(f"Cost: {result.cost:.2f}")
        print(f"Number of nodes created: {result.nodes_created_count}")

    def handle_run_benchmarks():
        """Run the benchmarks for the path finder and display the average results."""

        _, average_results = run_full_benchmark(
            adjacency_matrix=adjacency_matrix,
            node_coordinates=node_coordinates,
            slope_matrix=slope_matrix,
            kerb_ramps_matrix=kerb_ramps_matrix,
            sidewalk_width_matrix=sidewalk_width_matrix,
            test_case_count=TEST_CASE_COUNT,
            runs_per_test_case=RUNS_PER_TEST_CASE,
        )
        print(
            f"Average results of {TEST_CASE_COUNT} test cases ({RUNS_PER_TEST_CASE} runs each):"
        )
        print(average_results)

    (
        adjacency_matrix,
        node_coordinates,
        slope_matrix,
        kerb_ramps_matrix,
        sidewalk_width_matrix,
    ) = initialize_map().values()

    print("==" * 40)
    print("Welcome to Wheelchair Path Finder!")
    print("Please choose an option:")
    print(" 1. Run path finder")
    print(" 2. Run benchmarks")
    choice = input("\n>>> Your choice: ").strip()
    print("==" * 40)
    if choice == "1":
        handle_run_path_finder()
    elif choice == "2":
        handle_run_benchmarks()


if __name__ == "__main__":
    main()
