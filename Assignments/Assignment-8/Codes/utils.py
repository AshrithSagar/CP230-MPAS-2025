"""
utils.py \n
Utility functions
"""

import logging
import random
from collections import deque
from typing import Deque, Dict, List, Optional, Set, TypeVar

logger = logging.getLogger(__name__)


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
        _repr = "UndirectedGraph(\n"
        for node_id, node in sorted(self.nodes.items(), key=lambda x: x[0]):
            _repr += f"  {node_id}: {sorted([nbr.id for nbr in node.neighbors])}\n"
        _repr += ")"
        return _repr

    def __contains__(self, id: NodeType) -> bool:
        """
        Check if a node with the given id exists in the graph. \n
        :param id: The id of the node.
        :return: True if the node exists, False otherwise.
        """
        return id in self.nodes

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

    def to_dict(self) -> GraphType:
        """
        Convert the graph to a dictionary representation. \n
        :return: A dictionary mapping nodes to lists of neighbors.
        """
        graph_dict: GraphType = {}
        for node_id, node in sorted(self.nodes.items(), key=lambda x: x[0]):
            graph_dict[node_id] = sorted([nbr.id for nbr in node.neighbors])
        return graph_dict

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

    @classmethod
    def generate_random(
        cls, num_nodes: int, max_neighbors: int = 3, seed: int = 42
    ) -> "UndirectedGraph":
        """
        Generate a random undirected graph. \n
        :param num_nodes: Total number of nodes in the graph.
        :param max_neighbors: Max neighbors per node (edge density).
        :param seed: Random seed for reproducibility.
        :return: An UndirectedGraph object.
        """
        graph = cls()
        random.seed(seed)
        node_ids: List[NodeType] = list(range(num_nodes))

        for node_id in node_ids:
            graph.add_node(node_id)
            num_edges = random.randint(1, max_neighbors)
            possible_neighbors = [n for n in node_ids if n != node_id]
            neighbors = random.sample(possible_neighbors, k=num_edges)

            for nbr_id in neighbors:
                graph.add_edge(node_id, nbr_id)

        return graph


def bfs_fifo(
    graph: UndirectedGraph, start: NodeType, goal: NodeType, verbose: bool = False
) -> List[NodeType]:
    """
    Perform FIFO (breadth-first) search on an undirected graph. \n
    :param graph: The undirected graph to search.
    :param start: The starting node id.
    :param goal: The goal node id.
    :param verbose: If True, show detailed output.
    :return: A list of node ids in the order they were expanded.
    """
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING, format="%(message)s"
    )
    logger.info("Starting BFS FIFO search...")

    if start not in graph:
        logging.info("Start node not in graph!")
        return []
    elif goal not in graph:
        logging.info("Goal node not in graph!")
        return []

    explored: Set[NodeType] = set()
    queue: Deque[NodeType] = deque([start])
    expansion_order: List[NodeType] = []

    iteration = 0
    while queue:
        iteration += 1
        logging.info(f"Iteration {iteration}:")
        logging.info(f"  Active nodes: {list(queue)}")
        logging.info(f"  Explored nodes: {list(explored)}")
        node_id = queue.popleft()
        if node_id in explored:
            continue

        # Mark as explored
        explored.add(node_id)
        expansion_order.append(node_id)
        logging.info(f"  Expanding node: {node_id}")

        if node_id == goal:  # If this is the goal, stop
            logging.info("Goal found!")
            break

        # Enqueue all unseen neighbors
        additions = []
        for nbr_id in graph.get_neighbors(node_id):
            if nbr_id not in explored and nbr_id not in queue:
                queue.append(nbr_id)
                additions.append(nbr_id)
        logging.info(f"  Populating queue with: {additions}")

    return expansion_order
