"""
Main Entry Point for the Wheelchair Path Finder Application.
"""

import json
import logging
import math
import os

import pandas as pd
import numpy as np
from colorama import Fore, Style

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


WHEELCHAIR_SPEED_KM_PER_H = 4.0
WHEELCHAIR_SPEED_M_PER_S = WHEELCHAIR_SPEED_KM_PER_H * 1_000 / (60 * 60)

TEST_CASE_COUNT = 20
RUNS_PER_TEST_CASE = 5


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
        # Select start and goal locations
        print("\nAvailable locations:")
        for index, location in enumerate(adjacency_matrix.index, start=1):
            print(f"{Fore.MAGENTA}{index}.{Style.RESET_ALL} {location}")

        start_input_text = (
            f"\n{Fore.CYAN}>>>{Style.RESET_ALL} Enter start location ID: "
        )
        start_index = int(input(start_input_text).strip()) - 1
        start = adjacency_matrix.index[start_index]

        goal_input_text = f"{Fore.CYAN}>>>{Style.RESET_ALL} Enter goal location ID: "
        goal_index = int(input(goal_input_text).strip()) - 1
        goal = adjacency_matrix.index[goal_index]

        accessibility_input_text = f"{Fore.CYAN}>>>{Style.RESET_ALL} Consider wheelchair accessibility? (y/n): "
        consider_accessibility = input(accessibility_input_text).strip().lower() == "y"

        # Run the path finder
        path_finder = AStar(
            adjacency_matrix=adjacency_matrix,
            node_coordinates=node_coordinates,
            slope_matrix=slope_matrix,
            kerb_ramps_matrix=kerb_ramps_matrix,
            sidewalk_width_matrix=sidewalk_width_matrix,
        )
        result = path_finder.find_path(
            start, goal, consider_accessibility=consider_accessibility
        )

        # Calculate the estimated time
        seconds = result.distance / WHEELCHAIR_SPEED_M_PER_S
        hours = math.floor(seconds / (60 * 60))
        minutes = round((seconds % (60 * 60)) / 60)
        hours_text = f"{hours} hours " if hours else ""
        minutes_text = f"{minutes} minutes" if minutes else ""

        # Display the results
        print(f"\n{Fore.BLACK}{'=' * terminal_width}{Style.RESET_ALL}")
        print(
            f"Path: {Fore.MAGENTA}{f"  {Fore.CYAN}⮕{Fore.MAGENTA}  ".join(result.path)}{Style.RESET_ALL}"
        )
        print(
            f"Distance: {Fore.CYAN}{round(result.distance):,} meters{Style.RESET_ALL}"
        )
        print(f"Estimated time: {Fore.CYAN}{hours_text}{minutes_text}{Style.RESET_ALL}")

        print(f"\n{Fore.LIGHTBLACK_EX}Technical details:")
        print(f"  - Cost: {result.cost:,.2f}")
        print(f"  - Number of nodes created: {result.nodes_created_count}")
        print(
            f"  - Number of nodes explored: {result.nodes_explored_count}{Style.RESET_ALL}"
        )
        print(f"{Fore.BLACK}{'=' * terminal_width}{Style.RESET_ALL}")

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
        average_results_table = average_results.to_string(index=False, justify="center")
        print(
            f"\nAverage results ({TEST_CASE_COUNT} test cases, {RUNS_PER_TEST_CASE} runs each):"
        )
        print(f"{Fore.BLACK}{'=' * terminal_width}{Style.RESET_ALL}")
        print(average_results_table)
        print(f"{Fore.BLACK}{'=' * terminal_width}{Style.RESET_ALL}")

    # Set up the map
    (
        adjacency_matrix,
        node_coordinates,
        slope_matrix,
        kerb_ramps_matrix,
        sidewalk_width_matrix,
    ) = initialize_map().values()
    terminal_width = os.get_terminal_size().columns

    print(
        Fore.CYAN
        + """
____       _   _       _____ _           _           
|  _ \\ __ _| |_| |__   |  ___(_)_ __   __| | ___ _ __ 
| |_) / _` | __| '_ \\  | |_  | | '_ \\ / _` |/ _ \\ '__|
|  __/ (_| | |_| | | | |  _| | | | | | (_| |  __/ |   
|_|   \\__,_|\\__|_| |_| |_|   |_|_| |_|\\__,_|\\___|_|   
    """
    )
    print(Style.RESET_ALL + "\nWelcome to Wheelchair Path Finder!")
    while True:
        print("\nWhat would you like to do?")
        print(f" {Fore.MAGENTA}1.{Style.RESET_ALL} Run path finder")
        print(f" {Fore.MAGENTA}2.{Style.RESET_ALL} Run benchmarks")
        print(f" {Fore.MAGENTA}3.{Style.RESET_ALL} Exit")
        choice = input(f"\n{Fore.CYAN}>>>{Style.RESET_ALL} Your choice: ").strip()
        if choice == "1":
            handle_run_path_finder()
        elif choice == "2":
            handle_run_benchmarks()
        else:
            print(f"{Fore.MAGENTA}Thank you. Goodbye!{Style.RESET_ALL}")
            break


if __name__ == "__main__":
    main()
