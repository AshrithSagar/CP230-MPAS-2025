"""
utils.py \n
Utility functions
"""

from collections import deque
from typing import Deque, Dict, List, Optional, Set, TypeVar

NodeType = TypeVar("NodeType", str, int)
"""Type of the node in the graph"""


GraphType = Dict[NodeType, List[NodeType]]
"""
Type of the graph, which is a dictionary mapping nodes to a list of neighboring nodes.
The keys are nodes and the values are lists of neighbors.
"""


class Node:
    """A class representing a node in a graph."""

    def __init__(self, id: NodeType) -> None:
        """
        Initialize a node with a `id` and an empty list of neighbors. \n
        :param id: The id of the node.
        """
        self.id: NodeType = id
        self.neighbors: List["Node"] = []

    def __repr__(self) -> str:
        return f"Node({self.id})"

    def add_neighbor(self, node: "Node") -> None:
        """
        Add a neighbor to the node's list of neighbors. \n
        :param node: The neighbor node to add.
        """
        if node not in self.neighbors:
            self.neighbors.append(node)


class UndirectedGraph:
    """A class representing an undirected graph."""

    def __init__(self) -> None:
        """Initialize an empty graph with a dictionary to hold nodes."""
        self.nodes: Dict[NodeType, Node] = {}

    def __repr__(self) -> str:
        return f"UndirectedGraph({self.nodes})"

    def __contains__(self, id: NodeType) -> bool:
        """
        Check if a node with the given id exists in the graph. \n
        :param id: The id of the node.
        :return: True if the node exists, False otherwise.
        """
        return id in self.nodes

    @classmethod
    def from_dict(cls, graph_dict: GraphType) -> "UndirectedGraph":
        """
        Create an undirected graph from a dictionary. \n
        :param graph_dict: A dictionary mapping nodes to lists of neighbors.
        :return: An UndirectedGraph object.
        """
        graph = cls()
        for node_id, neighbors in graph_dict.items():
            for neighbor in neighbors:
                graph.add_edge(node_id, neighbor)
        return graph

    def add_node(self, id: NodeType) -> "Node":
        """
        Add a node to the graph. If the node already exists, return it. \n
        :param id: The id of the node.
        :return: The node object.
        """
        if id not in self.nodes:
            self.nodes[id] = Node(id)
        return self.nodes[id]

    def add_edge(self, id1: NodeType, id2: NodeType) -> None:
        """
        Add an edge between two nodes in the graph. \n
        :param id1: The id of the first node.
        :param id2: The id of the second node.
        """
        if id1 == id2:
            return
        node1 = self.add_node(id1)
        node2 = self.add_node(id2)
        node1.add_neighbor(node2)
        node2.add_neighbor(node1)  # Undirected graph

    def get_node(self, id: NodeType) -> Optional["Node"]:
        """
        Get a node from the graph by its id. \n
        :param id: The id of the node.
        :return: The node object if it exists, or None if it doesn't.
        """
        return self.nodes.get(id, None)

    def get_neighbors(self, id: NodeType) -> List[NodeType]:
        """
        Get the neighbors of a node. \n
        :param id: The id of the node.
        :return: A list of neighbor ids.
        """
        node = self.get_node(id)
        if node is None:
            return []
        return [nbr.id for nbr in node.neighbors]


def bfs_fifo(
    graph: UndirectedGraph, start: NodeType, goal: NodeType, verbose: bool = False
) -> List[NodeType]:
    """
    Perform FIFO (breadth-first) search on an undirected graph. \n
    :param graph: The undirected graph to search.
    :param start: The starting node id.
    :param goal: The goal node id.
    :param verbose: If True, print debug information.
    :return: A list of node ids in the order they were expanded.
    """
    if start not in graph:
        if verbose:
            print("Start node not in graph!")
        return []
    elif goal not in graph:
        if verbose:
            print("Goal node not in graph!")
        return []

    explored: Set[NodeType] = set()
    queue: Deque[NodeType] = deque([start])
    expansion_order: List[NodeType] = []

    while queue:
        node_id = queue.popleft()
        if node_id in explored:
            continue

        # Mark as explored
        explored.add(node_id)
        expansion_order.append(node_id)
        if verbose:
            print(f"Expanded: {node_id}")

        if node_id == goal:  # If this is the goal, stop
            if verbose:
                print("Goal found!")
            break

        # Enqueue all unseen neighbors
        for nbr_id in graph.get_neighbors(node_id):
            if nbr_id not in explored and nbr_id not in queue:
                queue.append(nbr_id)

    return expansion_order
