"""
utils.py \n
Utility functions
"""

from collections import deque


def bfs_fifo(graph, start, goal):
    """
    Perform FIFO (breadth-first) search on an undirected graph.

    :param graph: dict mapping nodes to list of neighbors
    :param start: start node
    :param goal: goal node
    :return: list of nodes in the order they were expanded
    """
    explored = set()
    queue = deque([start])
    expansion_order = []

    while queue:
        node = queue.popleft()
        if node in explored:
            continue

        # Mark as explored
        explored.add(node)
        expansion_order.append(node)
        print(f"Expanded: {node}")

        if node == goal:  # If this is the goal, stop
            print("Goal found!")
            break

        # Enqueue all unseen neighbors
        for nbr in graph.get(node, []):
            if nbr not in explored:
                queue.append(nbr)

    return expansion_order
