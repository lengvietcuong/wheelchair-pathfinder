"""
A* Algorithm Implementation for Wheelchair-Accessible Route Planning.
"""

import heapq
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Tuple, Set, Optional, TypedDict, Any

import numpy as np
import pandas as pd


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class NodeInfo(TypedDict):
    """Type definition for node information."""

    name: str
    coords: Tuple[float, float]


@dataclass(order=True)
class PrioritizedNode:
    """Node wrapper with priority for the priority queue."""

    priority: float
    node_id: str = field(compare=False)


def calculate_accessibility_costs(
    adjacency_matrix: pd.DataFrame,
    slope_df: pd.DataFrame,
    kerb_ramps_df: pd.DataFrame,
    sidewalk_width_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate accessibility-adjusted costs based on various accessibility features.
    
    Parameters:
        adjacency_matrix (pd.DataFrame): Base adjacency matrix with distances
        slope_df (pd.DataFrame): Matrix of slope values between nodes
        kerb_ramps_df (pd.DataFrame): Matrix indicating presence of kerb ramps (1) or absence (0)
        sidewalk_width_df (pd.DataFrame): Matrix of sidewalk widths between nodes
    
    Returns:
        pd.DataFrame: Accessibility-adjusted cost matrix
    """
    locations = adjacency_matrix.index.tolist()
    accessibility_df = pd.DataFrame(index=locations, columns=locations).fillna(np.inf)
    
    for i in locations:
        for j in locations:
            dist = adjacency_matrix.loc[i, j]
            if dist == np.inf or dist <= 0:
                continue
            
            slope = slope_df.loc[i, j]
            kerb = kerb_ramps_df.loc[i, j]
            width = sidewalk_width_df.loc[i, j]

            slope_factor = 1 + (slope / 5)
            kerb_factor = 2 if kerb == 0 else 1
            width_factor = max(1, 1.5 * (1.2 / width))

            accessibility_df.loc[i, j] = dist * slope_factor * kerb_factor * width_factor
            
    return accessibility_df


class AStarPathfinder:
    """
    A* pathfinding algorithm implementation for wheelchair-accessible navigation.

    This class implements the A* search algorithm to find optimal paths between
    locations, taking into account accessibility features like slopes, kerb ramps,
    and sidewalk widths.
    """

    def __init__(
        self,
        adjacency_matrix: pd.DataFrame,
        node_coordinates: Dict[str, Tuple[float, float]] = None,
        slope_matrix: pd.DataFrame = None,
        kerb_ramps_matrix: pd.DataFrame = None,
        sidewalk_width_matrix: pd.DataFrame = None,
    ):
        """
        Initialize the A* pathfinder.

        Parameters:
            adjacency_matrix (pd.DataFrame): The base adjacency matrix representing distances.
            node_coordinates (Dict[str, Tuple[float, float]], optional): Dictionary mapping node names to coordinates.
            slope_matrix (pd.DataFrame, optional): Matrix of slope values between nodes.
            kerb_ramps_matrix (pd.DataFrame, optional): Matrix indicating presence of kerb ramps (1) or absence (0).
            sidewalk_width_matrix (pd.DataFrame, optional): Matrix of sidewalk widths between nodes.
        """
        self.adjacency_matrix = adjacency_matrix
        self.node_coordinates = node_coordinates or {}
        self.slope_matrix = slope_matrix
        self.kerb_ramps_matrix = kerb_ramps_matrix
        self.sidewalk_width_matrix = sidewalk_width_matrix
        self.locations = list(adjacency_matrix.index)

        # Calculate accessibility costs if all accessibility matrices are provided
        if all(
            [
                slope_matrix is not None,
                kerb_ramps_matrix is not None,
                sidewalk_width_matrix is not None,
            ]
        ):
            self.accessibility_matrix = calculate_accessibility_costs(
                adjacency_matrix,
                slope_matrix,
                kerb_ramps_matrix,
                sidewalk_width_matrix,
            )
        else:
            self.accessibility_matrix = adjacency_matrix

    def euclidean_distance(self, node1: str, node2: str) -> float:
        """
        Calculate the Euclidean distance between two nodes.

        Parameters:
            node1 (str): First node name.
            node2 (str): Second node name.

        Returns:
            float: Euclidean distance between nodes, or 0 if coordinates unavailable.
        """
        if (
            not self.node_coordinates
            or node1 not in self.node_coordinates
            or node2 not in self.node_coordinates
        ):
            return 0.0

        x1, y1 = self.node_coordinates[node1]
        x2, y2 = self.node_coordinates[node2]
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    def straight_line_heuristic(self, node: str, goal: str) -> float:
        """
        Calculate the straight-line distance heuristic for A*.

        Parameters:
            node (str): Current node name.
            goal (str): Goal node name.

        Returns:
            float: Heuristic value (Euclidean distance).
        """
        return self.euclidean_distance(node, goal)

    def find_path(
        self,
        start: str,
        goal: str,
        heuristic_fn: Callable[[str, str], float] = None,
        use_accessibility: bool = True,
    ) -> Tuple[List[str], float, Dict[str, Any]]:
        """
        Find the optimal path from start to goal using A* algorithm.
        """
        if start not in self.locations:
            raise ValueError(f"Start location '{start}' not found in adjacency matrix")
        if goal not in self.locations:
            raise ValueError(f"Goal location '{goal}' not found in adjacency matrix")

        if heuristic_fn is None:
            heuristic_fn = self.straight_line_heuristic

        matrix = (
            self.accessibility_matrix if use_accessibility else self.adjacency_matrix
        )

        open_set: List[PrioritizedNode] = []
        # Track nodes in open set for efficient lookup
        open_set_nodes: Set[str] = set()
        closed_set: Set[str] = set()

        g_scores: Dict[str, float] = {node: float("inf") for node in self.locations}
        g_scores[start] = 0

        f_scores: Dict[str, float] = {node: float("inf") for node in self.locations}
        f_scores[start] = heuristic_fn(start, goal)

        parents: Dict[str, Optional[str]] = {node: None for node in self.locations}

        heapq.heappush(open_set, PrioritizedNode(f_scores[start], start))
        open_set_nodes.add(start)

        nodes_explored = 0

        while open_set:
            current_node = heapq.heappop(open_set).node_id

            # Skip if we've already processed this node
            if current_node in closed_set:
                continue

            open_set_nodes.remove(current_node)
            nodes_explored += 1

            if current_node == goal:
                path = self._reconstruct_path(parents, goal)
                path_cost = g_scores[goal]
                stats = {
                    "nodes_explored": nodes_explored,
                    "path_length": len(path),
                    "matrix_used": "accessibility" if use_accessibility else "base",
                }
                return path, path_cost, stats

            closed_set.add(current_node)

            for neighbor in self.locations:
                if (
                    neighbor == current_node
                    or matrix.loc[current_node, neighbor] == np.inf
                ):
                    continue
                if neighbor in closed_set:
                    continue

                tentative_g_score = (
                    g_scores[current_node] + matrix.loc[current_node, neighbor]
                )

                if tentative_g_score >= g_scores[neighbor]:
                    continue  # Not a better path

                # Update path information
                parents[neighbor] = current_node
                g_scores[neighbor] = tentative_g_score
                f_scores[neighbor] = tentative_g_score + heuristic_fn(neighbor, goal)

                # Always add to open set with updated priority
                if neighbor not in open_set_nodes:
                    open_set_nodes.add(neighbor)

                heapq.heappush(open_set, PrioritizedNode(f_scores[neighbor], neighbor))

        return (
            [],
            float("inf"),
            {"nodes_explored": nodes_explored, "error": "No path found"},
        )

    def _reconstruct_path(
        self, parents: Dict[str, Optional[str]], goal: str
    ) -> List[str]:
        """
        Reconstruct the path from start to goal using parent nodes.

        Parameters:
            parents (Dict[str, Optional[str]]): Dictionary of parent nodes.
            goal (str): Goal node name.

        Returns:
            List[str]: Path from start to goal as a list of node names.
        """
        path = [goal]
        current = goal

        while parents[current] is not None:
            current = parents[current]
            path.append(current)

        return path[::-1]

    def accessibility_heuristic(self, node: str, goal: str) -> float:
        """
        Enhanced heuristic that accounts for accessibility features.

        Parameters:
            node (str): Current node name.
            goal (str): Goal node name.

        Returns:
            float: Heuristic value incorporating accessibility.
        """
        base_distance = self.straight_line_heuristic(node, goal)
        if base_distance == 0:
            min_distance = np.min(
                self.adjacency_matrix.values[self.adjacency_matrix.values > 0]
            )
            base_distance = min_distance
        accessibility_factor = 1.0
        return base_distance * accessibility_factor


def extract_node_coordinates(
    nodes_dictionary: Dict[int, NodeInfo],
) -> Dict[str, Tuple[float, float]]:
    """
    Extract node coordinates from nodes dictionary.

    Parameters:
        nodes_dictionary (Dict[int, NodeInfo]): Dictionary of node information.

    Returns:
        Dict[str, Tuple[float, float]]: Dictionary mapping node names to coordinates.
    """
    return {
        node_data["name"]: node_data["coords"]
        for node_data in nodes_dictionary.values()
    }


def main():
    """Main function to test the A* implementation."""
    try:
        adjacency_matrix = pd.read_csv("data/adjacency_matrix.csv", index_col=0)
        adjacency_matrix = adjacency_matrix.replace("inf", np.inf)
    except Exception as error:
        logger.error(f"Failed to load adjacency matrix: {error}", exc_info=True)
        return

    # Load accessibility feature matrices
    try:
        slope_matrix = pd.read_csv("data/adjacency_matrix_slope.csv", index_col=0)
        kerb_ramps_matrix = pd.read_csv("data/adjacency_matrix_kerb_ramps.csv", index_col=0)
        sidewalk_width_matrix = pd.read_csv("data/adjacency_matrix_sidewalk_width.csv", index_col=0)
    except FileNotFoundError as error:
        logger.warning(f"Accessibility feature matrix not found: {error}. Using base adjacency matrix.")
        slope_matrix = None
        kerb_ramps_matrix = None
        sidewalk_width_matrix = None
    except Exception as error:
        logger.error(f"Error loading accessibility feature matrices: {error}", exc_info=True)
        slope_matrix = None
        kerb_ramps_matrix = None
        sidewalk_width_matrix = None

    pathfinder = AStarPathfinder(
        adjacency_matrix=adjacency_matrix,
        slope_matrix=slope_matrix,
        kerb_ramps_matrix=kerb_ramps_matrix,
        sidewalk_width_matrix=sidewalk_width_matrix,
    )

    # Set start and destination using the required values.
    start_location = "Noi Due Cafe"
    destination_location = "Robert Lehman Collection Library"

    all_locations = set(adjacency_matrix.index)
    if start_location not in all_locations:
        logger.error(
            f"Start location '{start_location}' not found in adjacency matrix."
        )
        return
    if destination_location not in all_locations:
        logger.error(
            f"Destination location '{destination_location}' not found in adjacency matrix."
        )
        return

    logger.info(
        f"Finding path from '{start_location}' to '{destination_location}' using basic heuristic..."
    )
    path, cost, stats = pathfinder.find_path(
        start_location, destination_location, use_accessibility=False
    )
    logger.info(f"Path found: {path}")
    logger.info(f"Cost: {cost:.2f}")
    logger.info(f"Stats: {stats}")

    if slope_matrix is not None and kerb_ramps_matrix is not None and sidewalk_width_matrix is not None:
        logger.info(
            f"Finding path from '{start_location}' to '{destination_location}' using accessibility costs..."
        )
        path, cost, stats = pathfinder.find_path(
            start_location, destination_location, use_accessibility=True
        )
        logger.info(f"Path found: {path}")
        logger.info(f"Cost: {cost:.2f}")
        logger.info(f"Stats: {stats}")


if __name__ == "__main__":
    main()
