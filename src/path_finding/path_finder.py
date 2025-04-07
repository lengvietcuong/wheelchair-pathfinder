from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, NamedTuple, Optional, Set, Tuple

import numpy as np
import pandas as pd


class SearchResult(NamedTuple):
    """Result of a path search operation.

    Attributes:
        path (List[str]): The list of nodes representing the path from the start to the goal.
        cost (float): The total cost associated with the found path.
        nodes_created (int): The number of nodes generated during the search.
    """

    path: List[str]
    cost: float
    nodes_created: int


@dataclass(order=True)
class Move:
    """Represents a move between nodes with priority for the frontier.

    Attributes:
        source (str): The identifier of the source node.
        destination (str): The identifier of the destination node.
        priority (float): The priority value used by the priority queue (f-score).
        index (int): Tie-breaker index for moves with equal priority.
    """

    source: str = field(compare=False)
    destination: str = field(compare=False)
    priority: float
    index: int = field(compare=False, default=0)


def calculate_accessibility_costs(
    adjacency_matrix: pd.DataFrame,
    slope_df: pd.DataFrame,
    kerb_ramps_df: pd.DataFrame,
    sidewalk_width_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate accessibility-adjusted costs based on various accessibility features.

    Args:
        adjacency_matrix (pd.DataFrame): Base adjacency matrix with distances between nodes.
        slope_df (pd.DataFrame): Matrix of slope percentages between nodes.
        kerb_ramps_df (pd.DataFrame): Matrix indicating presence (1) or absence (0) of kerb ramps.
        sidewalk_width_df (pd.DataFrame): Matrix of sidewalk widths (in meters) between nodes.

    Returns:
        pd.DataFrame: Accessibility-adjusted cost matrix combining distance, slope, kerb ramps, and width factors.
    """
    locations = adjacency_matrix.index.tolist()
    accessibility_df = pd.DataFrame(
        index=locations, columns=locations, dtype=float
    ).fillna(np.inf)

    for i in locations:
        for j in locations:
            distance = adjacency_matrix.loc[i, j]
            if distance == np.inf or distance <= 0:
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

            accessibility_df.loc[i, j] = (
                distance * slope_factor * kerb_factor * width_factor
            )

    return accessibility_df


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
        self._nodes_created: Set[str] = set()
        self._move_index = 0

        has_accessibility_features = (
            slope_matrix is not None
            and kerb_ramps_matrix is not None
            and sidewalk_width_matrix is not None
        )
        if has_accessibility_features:
            self._adjusted_adjacency_matrix = calculate_accessibility_costs(
                adjacency_matrix,
                slope_matrix,
                kerb_ramps_matrix,
                sidewalk_width_matrix,
            )
        else:
            self._adjusted_adjacency_matrix = None

    def _is_valid_move(self, source: str, destination: str) -> bool:
        """
        Check if moving from source to destination is valid.

        Args:
            source (str): Identifier of the current node.
            destination (str): Identifier of the neighbor node.

        Returns:
            bool: True if move is valid and neighbor has not been visited; False otherwise.
        """
        return (
            source in self._nodes
            and destination in self._nodes
            and source != destination
            and self._base_adjacency_matrix.loc[source, destination] != np.inf
            and destination not in self._came_from
        )

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
            self._nodes_created.add(neighbor)
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
        self._nodes_created = set()
        self._move_index = 0

    @abstractmethod
    def find_path(self, start: str, goal: str) -> SearchResult:
        """
        Abstract method to find a path from start to goal.

        Args:
            start (str): Identifier of the start node.
            goal (str): Identifier of the goal node.

        Returns:
            SearchResult: The result containing path, cost, and nodes created.
        """
        pass
