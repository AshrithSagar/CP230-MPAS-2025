"""
run.py \n
Run the BFS FIFO algorithm
"""

from utils import UndirectedGraph, bfs_fifo

if __name__ == "__main__":
    # Adjacency list
    G = {
        "I": ["I1", "I3", "I5"],
        "I1": ["I", "I2"],
        "I2": ["I1", "X", "Y"],
        "I3": ["I", "I4"],
        "I4": ["I3", "Z"],
        "I5": ["I", "I6"],
        "I6": ["I5", "W"],
        "G": ["I6", "W"],
    }
    graph = UndirectedGraph.from_dict(G)

    order = bfs_fifo(graph, start="I", goal="G", verbose=True)
    print("Expansion order:", order)
