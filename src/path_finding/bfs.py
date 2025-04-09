"""
Breadth-First Search (BFS) Pathfinding Algorithm Implementation.
"""

import logging
from collections import deque
from typing import Deque

from .path_finder import Move, PathFinder, SearchResult


logger = logging.getLogger(__name__)


class BFS(PathFinder):
    """
    Breadth-First Search pathfinding algorithm with optional wheelchair accessibility consideration.
    """

    def find_path(
        self,
        start: str,
        goal: str,
        consider_accessibility: bool = True,
    ) -> SearchResult:
        if start not in self._graph or goal not in self._graph:
            raise ValueError("Start or goal location not found")

        self._reset_states()
        # Choose appropriate adjacency matrix
        if consider_accessibility:
            logger.debug("Accessibility considered. Using adjusted adjacency matrix")
            adjacency_matrix = self._accessibility_adjacency_matrix
        else:
            logger.debug("Accessibility ignored. Using base adjacency matrix")
            adjacency_matrix = self._base_adjacency_matrix

        # Initialize a queue with a dummy move to the start node
        queue: Deque[Move] = deque([Move(source=None, destination=start)])
        while queue:
            move = queue.popleft()
            logger.debug(f"Exploring move from '{move.source}' to '{move.destination}'")
            if move.destination in self._came_from:  # Already visited
                logger.debug(f"Already visited '{move.destination}', skipping")
                continue

            self._came_from[move.destination] = move.source
            if move.destination == goal:
                logger.debug("Found goal node. Returing result")
                path = self._reconstruct_path(goal)
                cost = sum(
                    adjacency_matrix.loc[path[i], path[i + 1]]
                    for i in range(len(path) - 1)
                )
                return SearchResult(
                    path=path,
                    cost=cost,
                    nodes_created_count=len(self._nodes_created),
                )

            for new_move in self._expand(move.destination):
                queue.append(new_move)
                logger.debug(
                    f"Added move from '{new_move.source}' to '{new_move.destination}' to explore next"
                )

        logger.debug(f"No path found (explored {len(self._came_from)} nodes)")
        return SearchResult([], float("inf"), len(self._nodes_created))
