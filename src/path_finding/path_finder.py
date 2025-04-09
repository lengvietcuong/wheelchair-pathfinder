"""
Base Class for Pathfinding Algorithms.
"""

from abc import ABC, abstractmethod
from math import sqrt
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from .custom_types import Move, SearchResult


def calculate_accessibility_costs(
    adjacency_matrix: pd.DataFrame,
    slope_df: pd.DataFrame,
    kerb_ramps_df: pd.DataFrame,
    sidewalk_width_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, float]:
    """
    Calculate accessibility-accessibility costs based on various accessibility features.

    Args:
        adjacency_matrix (pd.DataFrame): Base adjacency matrix with distances between nodes.
        slope_df (pd.DataFrame): Matrix of slope percentages between nodes.
        kerb_ramps_df (pd.DataFrame): Matrix indicating presence (1) or absence (0) of kerb ramps.
        sidewalk_width_df (pd.DataFrame): Matrix of sidewalk widths (in meters) between nodes.

    Returns:
        Tuple[pd.DataFrame, float]: A tuple containing:
            - Accessibility-accessibility cost matrix combining distance, slope, kerb ramps, and width factors.
            - Minimum cost multiplier (compared to base costs).
    """
    locations = adjacency_matrix.index.tolist()
    accessibility_df = pd.DataFrame(
        index=locations, columns=locations, dtype=float
    ).fillna(np.inf)
    min_cost_multiplier = float("inf")

    for i in locations:
        for j in locations:
            distance = adjacency_matrix.loc[i, j]
            if np.isinf(distance) or distance <= 0:  # Valid value check
                continue

            slope = slope_df.loc[i, j]
            kerb = kerb_ramps_df.loc[i, j]
            width = sidewalk_width_df.loc[i, j]

            # Slope factor: Steeper slopes increase cost linearly (20% increase per 1% slope)
            slope_factor = 1 + (slope / 5)

            # Kerb factor: Missing kerb ramps double the cost
            kerb_factor = 2 if kerb == 0 else 1

            # Width factor: Narrower sidewalks increase cost inversely
            width_factor = max(1, 1.2 / width)

            cost_multiplier = slope_factor * kerb_factor * width_factor
            accessibility_df.loc[i, j] = distance * cost_multiplier
            min_cost_multiplier = min(min_cost_multiplier, cost_multiplier)

    if np.isinf(min_cost_multiplier):
        min_cost_multiplier = 1.0
    return accessibility_df, min_cost_multiplier


class PathFinder(ABC):
    """Base class for pathfinding algorithms with optional accessibility support."""

    def __init__(
        self,
        adjacency_matrix: pd.DataFrame,
        node_coordinates: Dict[str, Tuple[float, float]],
        slope_matrix: Optional[pd.DataFrame] = None,
        kerb_ramps_matrix: Optional[pd.DataFrame] = None,
        sidewalk_width_matrix: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Initialize the pathfinder with an adjacency matrix and optional accessibility feature matrices.

        Args:
            adjacency_matrix (pd.DataFrame): Base adjacency matrix with distances.
            node_coordinates (Dict[str, Tuple[float, float]]): Mapping of node identifiers to (x, y) coordinates.
            slope_matrix (Optional[pd.DataFrame]): Matrix of slope percentages between nodes.
            kerb_ramps_matrix (Optional[pd.DataFrame]): Matrix indicating presence of kerb ramps.
            sidewalk_width_matrix (Optional[pd.DataFrame]): Matrix of sidewalk widths between nodes.

        Returns:
            None
        """
        self._base_adjacency_matrix = adjacency_matrix
        self._node_coordinates = node_coordinates
        self._nodes = set(adjacency_matrix.index)

        self._came_from: Dict[str, Optional[str]] = {}
        self._nodes_created_count: Set[str] = set()
        self._move_index = 0

        self._accessibility_adjacency_matrix: Optional[pd.DataFrame] = None
        self._accessibility_cost_multiplier: float = 1.0

        has_accessibility_features = (
            slope_matrix is not None
            and kerb_ramps_matrix is not None
            and sidewalk_width_matrix is not None
        )
        if not has_accessibility_features:
            return
        # Account for accessibility features
        self._accessibility_adjacency_matrix, self._accessibility_cost_multiplier = (
            calculate_accessibility_costs(
                self._base_adjacency_matrix,
                slope_matrix,
                kerb_ramps_matrix,
                sidewalk_width_matrix,
            )
        )

    def _is_valid_move(self, source: str, destination: str) -> bool:
        """
        Check if moving from source to destination is valid.

        Args:
            source (str): Identifier of the current node.
            destination (str): Identifier of the neighbor node.

        Returns:
            bool: True if move is valid and neighbor has not been visited; False otherwise.
        """
        nodes_exist = source in self._nodes and destination in self._nodes
        path_exists = source != destination and not np.isinf(
            self._base_adjacency_matrix.loc[source, destination]
        )
        is_visited = destination in self._came_from

        return nodes_exist and path_exists and not is_visited

    def _expand(self, node: str) -> List[Move]:
        """
        Generate valid moves from the given node.

        Args:
            node (str): Identifier of the node to expand.

        Returns:
            List[Move]: List of possible moves from the node.
        """
        moves: List[Move] = []
        for neighbor in self._nodes:
            if not self._is_valid_move(node, neighbor):
                continue

            moves.append(
                Move(
                    source=node,
                    destination=neighbor,
                    priority=0,  # Will be set by caller
                    index=self._move_index,
                )
            )
            self._nodes_created_count.add(neighbor)
            self._move_index += 1

        return moves

    def _reconstruct_path(self, goal: str) -> List[str]:
        """
        Reconstruct the path from start to goal using the came_from map.

        Args:
            goal (str): Identifier of the goal node.

        Returns:
            List[str]: Ordered list of node identifiers from start to goal.
        """
        path: List[str] = [goal]
        current = goal

        while self._came_from.get(current) is not None:
            current = self._came_from[current]  # type: ignore
            path.append(current)

        return list(reversed(path))

    def _reset_states(self) -> None:
        """
        Reset internal search state for a new pathfinding operation.
        """
        self._came_from = {}
        self._nodes_created_count = set()
        self._move_index = 0

    def _get_euclidean_distance(self, source: str, destination: str) -> float:
        """
        Estimate cost from source to destination using Euclidean distance.

        This is an admissible heuristic that never overestimates the distance to the goal, making it suitable for finding optimal paths with A*.

        Args:
            source (str): Identifier of the source node.
            destination (str): Identifier of the destination node.

        Returns:
            float: Euclidean distance between the two nodes.
        """
        x1, y1 = self._node_coordinates[source]
        x2, y2 = self._node_coordinates[destination]
        return sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * 100

    @abstractmethod
    def find_path(
        self,
        start: str,
        goal: str,
        consider_accessibility: bool = True,
    ) -> SearchResult:
        """
        Abstract method to find a path from start to goal.

        Args:
            start (str): Identifier of the start node.
            goal (str): Identifier of the goal node.
            consider_accessibility (bool): Whether to consider wheelchair accessibility when calculating the cost and heuristic. Defaults to True.

        Returns:
            SearchResult: The result containing path, cost, and nodes created.
        """
        pass
