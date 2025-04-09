"""
A* Pathfinding Algorithm Implementation.
"""

import heapq
import logging
from typing import Dict, List

from .custom_types import Move, SearchResult
from .path_finder import PathFinder


logger = logging.getLogger(__name__)


class AStar(PathFinder):
    """A* pathfinding algorithm with optional wheelchair accessibility consideration."""

    def find_path(
        self,
        start: str,
        goal: str,
        consider_accessibility: bool = True,
    ) -> SearchResult:
        if start not in self._graph or goal not in self._graph:
            raise ValueError("Start or goal location not found")

        self._reset_states()
        # Choose appropriate adjacency matrix and heuristic cost multiplier
        if consider_accessibility:
            logger.debug(
                f"Accessibility considered. Using adjusted adjacency matrix and cost multiplier ({self._accessibility_cost_multiplier}x)"
            )
            adjacency_matrix = self._accessibility_adjacency_matrix
            heuristic_cost_multiplier = self._accessibility_cost_multiplier
        else:
            logger.debug(
                "Accessibility ignored. Using base adjacency matrix and cost multiplier"
            )
            adjacency_matrix = self._base_adjacency_matrix
            heuristic_cost_multiplier = 1.0

        # Track known costs
        g_scores: Dict[str, float] = {node: float("inf") for node in self._graph}
        g_scores[start] = 0.0
        # Initialize frontier with a dummy move to the start node
        frontier: List[Move] = [Move(source=None, destination=start)]
        while frontier:
            move = heapq.heappop(frontier)
            logger.debug(f"Exploring move from '{move.source}' to '{move.destination}'")

            # Skip if node is already visited
            if move.destination in self._came_from:
                logger.debug(f"Already visited '{move.destination}', skipping")
                continue

            # Record where we came from
            self._came_from[move.destination] = move.source

            if move.destination == goal:  # At goal node
                logger.debug("Found goal node. Returing result")
                return SearchResult(
                    path=self._reconstruct_path(goal),
                    cost=g_scores[goal],
                    nodes_created_count=len(self._nodes_created),
                )

            # Add neighbors to frontier
            for new_move in self._expand(move.destination):
                neighbor = new_move.destination
                # Calculate the known cost to reach this neighbor (if going through the current node)
                g_score = (
                    g_scores[move.destination]
                    + adjacency_matrix.loc[move.destination, neighbor]
                )
                if g_score >= g_scores[neighbor]:  # Already found a better path
                    continue

                g_scores[neighbor] = g_score  # Record new best cost
                # Calculate the priority score as known_cost + estimated_cost
                f_score = (
                    g_score
                    + self._get_euclidean_distance(neighbor, goal)
                    * heuristic_cost_multiplier
                )
                new_move.priority = f_score
                heapq.heappush(frontier, new_move)
                logger.debug(
                    f"Added move from '{new_move.source}' to '{new_move.destination}' to explore next. Priority: {new_move.priority}"
                )

        logger.debug(f"No path found (explored {len(self._came_from)} nodes)")
        return SearchResult(
            path=[], cost=float("inf"), nodes_created_count=len(self._nodes_created)
        )
