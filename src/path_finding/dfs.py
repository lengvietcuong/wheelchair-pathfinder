"""
Depth-First Search (DFS) Pathfinding Algorithm Implementation.
"""

import logging
from typing import List

from .custom_types import Move, SearchAction, SearchResult, SearchStep
from .path_finder import PathFinder


logger = logging.getLogger(__name__)


class DFS(PathFinder):
    """
    Depth-First Search pathfinding algorithm with optional wheelchair accessibility consideration.
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

        # Initialize a stack with a dummy move to the start node
        stack: List[Move] = [Move(source=None, destination=start)]
        # Record start node being added to frontier
        self._steps.append(SearchStep(action=SearchAction.ADD_TO_FRONTIER, node=start))
        
        while stack:
            move = stack.pop()
            logger.debug(f"Exploring move from '{move.source}' to '{move.destination}'")
            # Record node being explored
            self._steps.append(SearchStep(action=SearchAction.EXPLORE, node=move.destination))
            
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
                    steps=self._steps,
                )

            for new_move in self._expand(move.destination):
                stack.append(new_move)
                # Record node being added to frontier
                self._steps.append(SearchStep(action=SearchAction.ADD_TO_FRONTIER, node=new_move.destination))
                logger.debug(
                    f"Added move from '{new_move.source}' to '{new_move.destination}' to explore next"
                )

        logger.debug(f"No path found (explored {len(self._came_from)} nodes)")
        return SearchResult(
            path=[], 
            cost=float("inf"), 
            nodes_created_count=len(self._nodes_created),
            steps=self._steps,
        )
