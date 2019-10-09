from collections import defaultdict
from heapq import heapify, heappop, heappush
import networkx as nx
from AlgorithmE import *
import dijkstra3 as dijks


class Graph:
    def __init__(self):
        self.adj_list = defaultdict(list)

    def add_edge(self, from_node, to_node, cost, directed=False):
        """
        Adds an edge in the graph between from_node and to_node

        Parameters
        ----------
        from_node : Outward edge from this node
        to_node : Incoming edge to this node
        cost: Cost of traversing this edge
        directed: True if the edge is uni-directional, False if its bi-directional

        """

        if cost < 0: raise ValueError("Costs must be non-negative for dijkstra to work")

        self.adj_list[from_node].append((to_node, cost))
        if not directed:
            self.add_edge(to_node, from_node, cost, True)


def dijkstra(graph, source, destination):
    """
    Returns the shortest distance in the graph between source and destination nodes
    Returns None if source and destination are not connected or does not exist

    Parameters
    ----------
    graph : Graph object containing the nodes and edges
    source: Source node
    destination: Destination node

    Complexity
    ----------
    O((E + V) log(V)) roughly
    E : Number of edges in the graph
    V : Number of nodes in the graph

    """

    distance = {source: 0}
    queue = [(distance[source], source)]
    heapify(queue)

    while queue:
        (d, u) = heappop(queue)
        #if u == destination: return distance

        for v, w in graph.adj_list[u]:
            if (v not in distance) or ((d + w) < distance[v]):
                distance[v] = d + w
                heappush(queue, (distance[v], v))

    return distance
if __name__ == "__main__":
    graph = Graph()
    graph.add_edge("Redwood City", "Emerald Hills", 75)
    graph.add_edge("Redwood City", "Menlo Park", 100)
    graph.add_edge("Menlo Park", "Stanford", 50)
    graph.add_edge("Stanford", "Portola Valley", 100)
    graph.add_edge("Emerald Hills", "Portola Valley", 80)
    graph.add_edge("Emerald Hills", "Woodside", 50)
    graph.add_edge("Menlo Park", "Woodside", 200)


    distance = dijkstra(graph,"Redwood City","Woodside")

    G = nx.grid_2d_graph(40, 40)
    # Add weight 1 in each edge of the graph
    for i in G._adj:
        for j in G._adj[i]:
            G.add_edge(i, j, weight=1)

    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default',
                                                       label_attribute='old_labels')
    start = time.time()
    print("hello1")
    graph2 = Graph()
    for key, value in G._adj.items():
        for j in value:
            graph2.add_edge(key,j,1)

    distance2 = dijkstra(graph2, 0, 1)
    end = time.time()
    print((end - start) / 60, "min")


    start = time.time()
    print("hello2")
    test = Graphs()
    edges, adj = data_format(G._adj)
    graph = dijks.Graph(edges)
    temp_path, temp_distances = graph.dijkstra(0, 1)

    end = time.time()
    print((end - start) / 60, "min")
    print()
    # assert dijkstra(graph, "Stanford", "Woodside") == 230
    # assert dijkstra(graph, "Menlo Park", "Sunnyvale") == None