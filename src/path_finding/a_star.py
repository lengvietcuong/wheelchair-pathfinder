"""
A* Pathfinding Algorithm Implementation.
"""

import heapq
import logging
import math
from typing import Dict, List

from .path_finder import AccessibilitySetting, Move, PathFinder, SearchResult


logger = logging.getLogger(__name__)


class AStar(PathFinder):
    """A* pathfinding algorithm with optional wheelchair accessibility consideration."""

    def get_euclidean_distance(self, source: str, destination: str) -> float:
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
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def find_path(
        self,
        start: str,
        goal: str,
        accessibility: AccessibilitySetting = AccessibilitySetting.NONE,
    ) -> SearchResult:
        """
        Find the optimal path from start to goal using A* search.

        Args:
            start (str): Identifier of the start node.
            goal (str): Identifier of the goal node.
            accessibility (AccessibilitySetting): How to consider accessibility features.

        Returns:
            SearchResult: Contains the found path, total cost, and number of nodes created.
        """
        if start not in self._nodes or goal not in self._nodes:
            raise ValueError("Start or goal location not found")

        self._reset_states()
        # Track known costs
        g_scores: Dict[str, float] = {node: float("inf") for node in self._nodes}
        g_scores[start] = 0.0

        # Choose appropriate adjacency matrix
        if (
            accessibility != AccessibilitySetting.NONE
            and self._accessibility_adjacency_matrix is not None
        ):
            logger.debug("Using adjusted adjacency matrix for accessibility.")
            adjacency_matrix = self._accessibility_adjacency_matrix
        else:
            logger.debug("Using base adjacency matrix.")
            adjacency_matrix = self._base_adjacency_matrix

        # Choose appropriate cost multiple for the heuristic
        if accessibility == AccessibilitySetting.COST_AND_HEURISTIC:
            logger.debug("Using cost and heuristic for accessibility.")
            heuristic_cost_multiplier = self._accessibility_cost_multiplier
        else:
            logger.debug("Using heuristic only for accessibility.")
            heuristic_cost_multiplier = 1.0

        frontier: List[Move] = []
        # Initialize frontier with the start node
        heapq.heappush(
            frontier,
            Move(source=None, destination=start),
        )
        self._move_index += 1

        while frontier:
            move = heapq.heappop(frontier)
            logger.debug(f"Exploring move from '{move.source}' to '{move.destination}'")
            if move.destination in self._came_from:
                logger.debug(f"Already visited '{move.destination}', skipping.")
                continue

            self._came_from[move.destination] = move.source
            if move.destination == goal:
                return SearchResult(
                    path=self._reconstruct_path(goal),
                    cost=g_scores[goal],
                    nodes_created=len(self._nodes_created),
                )

            for new_move in self._expand(move.destination):
                neighbor = new_move.destination
                g_score = (
                    g_scores[move.destination]
                    + adjacency_matrix.loc[move.destination, neighbor]
                )
                if g_score >= g_scores[neighbor]:
                    continue

                g_scores[neighbor] = g_score
                # Calculate the priority score as known_cost + estimated_cost
                f_score = (
                    g_score
                    + self.get_euclidean_distance(neighbor, goal)
                    * heuristic_cost_multiplier
                )
                new_move.priority = f_score
                heapq.heappush(frontier, new_move)
                logger.debug(
                    f"Added move from '{new_move.source}' to '{new_move.destination}' to explore next"
                )

        return SearchResult([], float("inf"), len(self._nodes_created))
