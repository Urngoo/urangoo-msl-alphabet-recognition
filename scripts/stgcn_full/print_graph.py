import matplotlib.pyplot as plt
import numpy as np
from graph_hand import Graph

g = Graph(strategy="spatial")
A = g.A  # shape (3, 21, 21)

for i in range(A.shape[0]):
    plt.figure(figsize=(6,5))
    plt.imshow(A[i], cmap='viridis')
    plt.title(f"Adjacency Matrix - Subset {i}")
    plt.xlabel("Node Index")
    plt.ylabel("Node Index")
    plt.colorbar()
    plt.xticks(range(21))
    plt.yticks(range(21))
    plt.tight_layout()
    plt.show()