"""
graph.py
Graph data structure
"""

from typing import Dict, List, Union


class Vertex:
    def __init__(self, id: str) -> None:
        self.id = id
        self.neighbors: List[Vertex] = []

    def add_neighbor(self, vertex: "Vertex") -> None:
        self.neighbors.append(vertex)

    def remove_neighbor(self, vertex: "Vertex") -> None:
        if vertex in self.neighbors:
            self.neighbors.remove(vertex)

    def __str__(self) -> str:
        return self.id

    def __repr__(self) -> str:
        return f"Vertex({self.id})"

    def __eq__(self, other: "Vertex") -> bool:
        return self.id == other.id

    def __ne__(self, other: "Vertex") -> bool:
        return self.id != other.id

    def __hash__(self) -> int:
        return hash(self.id)


class Edge:
    def __init__(
        self, node1: Vertex, node2: Vertex, weight: Union[int, None] = None
    ) -> None:
        self.node1 = node1
        self.node2 = node2
        self.weight = weight

    def __str__(self) -> str:
        if self.weight is not None:
            return f"{self.node1} -[{self.weight}]-> {self.node2}"
        return f"{self.node1} -> {self.node2}"

    def __repr__(self) -> str:
        if self.weight is not None:
            return f"Edge({self.node1}, {self.node2}, {self.weight})"
        return f"Edge({self.node1}, {self.node2})"

    def __eq__(self, other: "Edge") -> bool:
        return (
            self.node1 == other.node1
            and self.node2 == other.node2
            and self.weight == other.weight
        )

    def __ne__(self, other: "Edge") -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash((self.node1, self.node2, self.weight))


class Graph:
    def __init__(self, directed: bool = False, weighted: bool = False) -> None:
        self.nodes: Dict[Vertex, List[Edge]] = {}
        self.directed = directed
        self.weighted = weighted

    def add_node(self, vertex: Vertex) -> None:
        if vertex not in self.nodes:
            self.nodes[vertex] = []

    def remove_node(self, vertex: Vertex) -> None:
        if vertex in self.nodes:
            del self.nodes[vertex]
            for edges in self.nodes.values():
                edges[:] = [edge for edge in edges if edge.node2 != vertex]

    def add_edge(
        self, node1: Vertex, node2: Vertex, weight: Union[int, None] = None
    ) -> None:
        if node1 in self.nodes and node2 in self.nodes:
            edge = Edge(node1, node2, weight if self.weighted else None)
            self.nodes[node1].append(edge)
            if not self.directed:
                reverse_edge = Edge(node2, node1, weight if self.weighted else None)
                self.nodes[node2].append(reverse_edge)

    def remove_edge(self, node1: Vertex, node2: Vertex) -> None:
        if node1 in self.nodes and node2 in self.nodes:
            self.nodes[node1] = [
                edge for edge in self.nodes[node1] if edge.node2 != node2
            ]
            if not self.directed:
                self.nodes[node2] = [
                    edge for edge in self.nodes[node2] if edge.node2 != node1
                ]

    def get_neighbors(self, vertex: Vertex) -> List[Vertex]:
        return [edge.node2 for edge in self.nodes[vertex]]

    def get_edges(self, vertex: Vertex) -> List[Edge]:
        return self.nodes[vertex]

    def __str__(self) -> str:
        return str(self.nodes)

    def __repr__(self) -> str:
        graph_type = "Directed" if self.directed else "Undirected"
        graph_weight = "Weighted" if self.weighted else "Unweighted"
        graph_str = f"{graph_type} {graph_weight} Graph:\n"
        for node, edges in self.nodes.items():
            edges_str = ", ".join(str(edge) for edge in edges)
            graph_str += f"  {node}: {edges_str}\n"
        return graph_str if self.nodes else f"{graph_type} {graph_weight} Graph (empty)"
