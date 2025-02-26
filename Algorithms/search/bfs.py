"""
bfs.py
Breadth-first search algorithm
"""

from queue import Queue
from typing import Dict, List, Union

from utils.graph import Graph, Vertex


def BFS(graph: Graph, start: Union[str, Vertex]) -> List[Vertex]:
    if isinstance(start, str):
        start = graph[start]
    visited: Dict[Vertex, bool] = {vertex: False for vertex in graph.vertices}
    queue = Queue()
    queue.put(start)
    visited[start] = True
    traversal: List[Vertex] = [start]
    while not queue.empty():
        vertex = queue.get()
        for neighbor in vertex.neighbors:
            if not visited[neighbor]:
                visited[neighbor] = True
                queue.put(neighbor)
                traversal.append(neighbor)
    return traversal
