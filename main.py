"""
Main Entry Point for the Wheelchair Path Finder Application.
"""

import logging
import math
import os
from pathlib import Path

from colorama import Fore, Style
from PIL import Image
from tabulate import tabulate

from map_creation.initialize_map import initialize_map
from path_finding.a_star import AStar
from path_finding.benchmark import run_full_benchmark


MAP_FILE_PATH = Path("map/map.png")

WHEELCHAIR_SPEED_KM_PER_H = 4.0
WHEELCHAIR_SPEED_M_PER_S = WHEELCHAIR_SPEED_KM_PER_H * 1_000 / (60 * 60)

TEST_CASE_COUNT = 20
RUNS_PER_TEST_CASE = 5

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    def handle_run_path_finder():
        """Take user input for start and goal locations, run the path finder, and display the outputs."""
        # Show the user all available locations
        headers = [
            f"{Fore.CYAN}{column}{Style.RESET_ALL}"
            for column in ("ID", "Location", "Accessibility features")
        ]
        locations_table = []
        for index, location in enumerate(adjacency_matrix.index, start=1):
            features = []
            if node_features.loc[location, "has_accessible_restroom"]:
                features.append("restroom")
            if node_features.loc[location, "has_accessible_parking"]:
                features.append("parking")
            if node_features.loc[location, "has_accessible_entrance"]:
                features.append("entrance")
            if node_features.loc[location, "has_rest_area"]:
                features.append("rest area")
            locations_table.append(
                [
                    f"{Fore.MAGENTA}{index}{Style.RESET_ALL}",
                    location,
                    ", ".join(features),
                ]
            )
        print(
            tabulate(
                locations_table, headers=headers, tablefmt="plain", numalign="left"
            )
        )

        # Get user input for start and goal locations
        start_input_text = (
            f"\n{Fore.CYAN}>>>{Style.RESET_ALL} Enter start location ID: "
        )
        start_index = int(input(start_input_text).strip()) - 1
        start = adjacency_matrix.index[start_index]

        goal_input_text = f"{Fore.CYAN}>>>{Style.RESET_ALL} Enter goal location ID: "
        goal_index = int(input(goal_input_text).strip()) - 1
        goal = adjacency_matrix.index[goal_index]

        # Ask if the user wants to consider wheelchair accessibility
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

        # Calculate the estimated time based on the distance
        seconds = result.distance / WHEELCHAIR_SPEED_M_PER_S
        hours = math.floor(seconds / (60 * 60))
        minutes = round((seconds % (60 * 60)) / 60)
        hours_text = f"{hours} hours " if hours else ""
        minutes_text = f"{minutes} minutes" if minutes else ""

        # Display the results
        print(f"\n{Fore.BLACK}{'=' * terminal_width}{Style.RESET_ALL}")
        print(
            f"Path: {Fore.MAGENTA}{f'  {Fore.CYAN}⮕{Fore.MAGENTA}  '.join(result.path)}{Style.RESET_ALL}"
        )
        print(
            f"Distance: {Fore.CYAN}{round(result.distance):,} meters{Style.RESET_ALL}"
        )
        print(f"Estimated time: {Fore.CYAN}{hours_text}{minutes_text}{Style.RESET_ALL}")

        print(f"\n{Fore.LIGHTBLACK_EX}Technical details:")
        print(f"  - Accessibility cost: {result.accessibility_cost:,.2f}")
        print(f"  - Number of nodes created: {result.nodes_created_count}")
        print(
            f"  - Number of nodes explored: {result.nodes_explored_count}{Style.RESET_ALL}"
        )
        print(f"{Fore.BLACK}{'=' * terminal_width}{Style.RESET_ALL}")

    def handle_run_benchmark():
        """Run the benchmark for the path finder and display the average results."""
        # Run the benchmark
        _, average_results = run_full_benchmark(
            adjacency_matrix=adjacency_matrix,
            node_coordinates=node_coordinates,
            slope_matrix=slope_matrix,
            kerb_ramps_matrix=kerb_ramps_matrix,
            sidewalk_width_matrix=sidewalk_width_matrix,
            test_case_count=TEST_CASE_COUNT,
            runs_per_test_case=RUNS_PER_TEST_CASE,
        )
        # Format and print the average results table
        headers = [
            f"{Fore.CYAN}{column}{Style.RESET_ALL}"
            for column in average_results.columns
        ]
        print(
            f"\nAverage results ({TEST_CASE_COUNT} test cases, {RUNS_PER_TEST_CASE} runs each):"
        )
        print(f"{Fore.BLACK}{'=' * terminal_width}{Style.RESET_ALL}")
        print(tabulate(average_results.values, headers=headers, tablefmt="plain"))
        print(f"{Fore.BLACK}{'=' * terminal_width}{Style.RESET_ALL}")

    def handle_view_map():
        """Open the map image using PIL (Pillow)."""
        print(f"{Fore.LIGHTBLACK_EX}Opening image...{Style.RESET_ALL}")
        image = Image.open(MAP_FILE_PATH)
        image.show(title="Map")
        print(f"{Fore.CYAN}Image opened!{Style.RESET_ALL}")

    # Set up the map
    (
        adjacency_matrix,
        node_coordinates,
        slope_matrix,
        kerb_ramps_matrix,
        sidewalk_width_matrix,
        node_features,
    ) = initialize_map().values()
    terminal_width = os.get_terminal_size().columns

    # Display greeting message and action menu
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
        print(f" {Fore.MAGENTA}2.{Style.RESET_ALL} Run benchmark")
        print(f" {Fore.MAGENTA}3.{Style.RESET_ALL} View map")
        print(f" {Fore.MAGENTA}4.{Style.RESET_ALL} Exit")
        choice = input(f"\n{Fore.CYAN}>>>{Style.RESET_ALL} Your choice: ").strip()
        try:
            if choice == "1":
                handle_run_path_finder()
            elif choice == "2":
                handle_run_benchmark()
            elif choice == "3":
                handle_view_map()
            else:
                print(f"{Fore.MAGENTA}Thank you. Goodbye!{Style.RESET_ALL}")
                break
        except Exception as error:
            print(f"{Fore.RED}Error: {error}{Style.RESET_ALL}")
            print(f"{Fore.LIGHTBLACK_EX}Please try again.{Style.RESET_ALL}")


if __name__ == "__main__":
    main()
