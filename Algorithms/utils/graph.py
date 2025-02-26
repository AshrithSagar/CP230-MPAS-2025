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

    def __eq__(self, other: Union[str, "Vertex"]) -> bool:
        if isinstance(other, str):
            return self.id == other
        return self.id == other.id

    def __ne__(self, other: "Vertex") -> bool:
        return self.id != other.id

    def __hash__(self) -> int:
        return hash(self.id)


class Edge:
    def __init__(
        self,
        vertex1: Union[str, Vertex],
        vertex2: Union[str, Vertex],
        weight: Union[int, None] = None,
    ) -> None:
        if isinstance(vertex1, str):
            vertex1 = Vertex(vertex1)
        if isinstance(vertex2, str):
            vertex2 = Vertex(vertex2)
        self.vertex1 = vertex1
        self.vertex2 = vertex2
        self.weight = weight

    def __str__(self) -> str:
        if self.weight is not None:
            return f"{self.vertex1} -[{self.weight}]-> {self.vertex2}"
        return f"{self.vertex1} -> {self.vertex2}"

    def __repr__(self) -> str:
        if self.weight is not None:
            return f"Edge({self.vertex1}, {self.vertex2}, {self.weight})"
        return f"Edge({self.vertex1}, {self.vertex2})"

    def __eq__(self, other: "Edge") -> bool:
        return (
            self.vertex1 == other.vertex1
            and self.vertex2 == other.vertex2
            and self.weight == other.weight
        )

    def __ne__(self, other: "Edge") -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash((self.vertex1, self.vertex2, self.weight))


class Graph:
    def __init__(self, directed: bool = False, weighted: bool = False) -> None:
        self.vertices: Dict[Vertex, List[Edge]] = {}
        self.directed = directed
        self.weighted = weighted

    def add_vertex(self, vertex: Union[str, Vertex]) -> None:
        if isinstance(vertex, str):
            vertex = Vertex(vertex)
        if vertex not in self.vertices:
            self.vertices[vertex] = []

    def remove_vertex(self, vertex: Vertex) -> None:
        if vertex in self.vertices:
            del self.vertices[vertex]
            for edges in self.vertices.values():
                edges[:] = [edge for edge in edges if edge.vertex2 != vertex]

    def add_edge(
        self, vertex1: Vertex, vertex2: Vertex, weight: Union[int, None] = None
    ) -> None:
        if vertex1 in self.vertices and vertex2 in self.vertices:
            edge = Edge(vertex1, vertex2, weight if self.weighted else None)
            self.vertices[vertex1].append(edge)
            if not self.directed:
                reverse_edge = Edge(vertex2, vertex1, weight if self.weighted else None)
                self.vertices[vertex2].append(reverse_edge)

    def remove_edge(self, vertex1: Vertex, vertex2: Vertex) -> None:
        if vertex1 in self.vertices and vertex2 in self.vertices:
            self.vertices[vertex1] = [
                edge for edge in self.vertices[vertex1] if edge.vertex2 != vertex2
            ]
            if not self.directed:
                self.vertices[vertex2] = [
                    edge for edge in self.vertices[vertex2] if edge.vertex2 != vertex1
                ]

    def get_neighbors(self, vertex: Vertex) -> List[Vertex]:
        return [edge.vertex2 for edge in self.vertices[vertex]]

    def get_edges(self, vertex: Vertex) -> List[Edge]:
        return self.vertices[vertex]

    def print(self) -> None:
        print(self.__repr__())

    def __str__(self) -> str:
        return str(self.vertices)

    def __repr__(self) -> str:
        graph_type = "Directed" if self.directed else "Undirected"
        graph_weight = "Weighted" if self.weighted else "Unweighted"
        graph_str = f"{graph_type} {graph_weight} Graph:\n"
        for vertex, edges in self.vertices.items():
            edges_str = ", ".join(str(edge) for edge in edges)
            graph_str += f"  {vertex}: {edges_str}\n"
        return (
            graph_str if self.vertices else f"{graph_type} {graph_weight} Graph (empty)"
        )

    def __getitem__(self, vertex: Union[str, Vertex]) -> List[Edge]:
        if isinstance(vertex, str):
            vertex = Vertex(vertex)
        return self.vertices[vertex]
