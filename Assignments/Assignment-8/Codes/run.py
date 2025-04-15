"""
run.py \n
Run the BFS FIFO algorithm
"""

from utils import UndirectedGraph, bfs_fifo

if __name__ == "__main__":
    graph = UndirectedGraph.generate_random(num_nodes=20, max_neighbors=3, seed=25)
    print("Graph:", graph)

    order = bfs_fifo(graph, start=0, goal=11, verbose=True)
    print("Expansion order:", order)
