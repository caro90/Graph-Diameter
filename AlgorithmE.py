from RangeTree import *
import numpy as np
from dijkstra3 import *
from binaryTrees import *


def data_format(graph):
    """
    Transforms the weight on the edges from networkx library, in a form readable for Dijkstra function
    Parameters:
    ----------
    graph: Graph
        A Graph object as it is created in networkx.

    Returns:
    --------
    edges2: list of lists
        The edges drawn from networkx library

    adj: dict
        The adjacency list of the given graph along with the weights on each edge
    """

    edges2 = []
    for i in (graph._adj):
        for j in graph._adj[i]:
            edges = [i, j, next(iter(graph._adj[i][j].values()))]
            if edges2 == []:
                edges2 = [edges[:]]
            else:
                edges2.append(edges)

    adj = graph._adj
    return edges2, adj


def algorithm_E(initial_graph, tw, portals_of_A, set_A, set_B):
    """
    # TODO: write summary of the algorithm
    Given a graph G and a skew k-separator tree, the algorithm computes the eccentricity e(v) of every
    vertex v in V(G). We write X = union(L,Z) and Y = union(R,Z)

    Parameters:
    ----------
        :


    Returns:
    --------


    """

    # TODO: not be dependent on networkx, change variables, eg initial_graph._node
    rangeTree = RangeTree()
    distances = []

    # edges: the edges of the initial_graph in readable format for Dijkstra
    edges, adj = data_format(initial_graph)
    # [E1] Base case of the recursion:
    # if n\ln(n) < 4tw(tw+1) then find all distances using Dijkstra's Algorithm and terminate
    n = len(initial_graph._node)
    if (n / np.log(n)) < 4*tw*(tw+1):
        # Graph is dijkstra's class
        graph = Graph(edges)
        for i in initial_graph._node:
            # graph.dijkstra(source, destination). The path from source to destination will be saved on path variable
            temp_path, temp_distances = graph.dijkstra(i, 0)
            distances.append(temp_distances)

    del temp_path, temp_distances
    # TODO: distances? eccentricities?
    #return

    print()

    # [E2] Distances from separator:
    #  Compute d(z,v) for every z in Z and every v in V(G), using k applications of Dijkstra's algorithm.
    distances_from_separator = {}
    count = 0
    for i in portals_of_A:
        temp_path, temp_distances = graph.dijkstra(i, 0)
        distances_from_separator[i] = temp_distances
        count += 1

    # [E3] Add shortcuts:
    # For each pair z,z' in Z, add the edge zz' to G, weighted by d(z,z'). Remove duplicates, retain shortest
    temp = list(portals_of_A)
    # Calculation of the pairs of the separator nodes, zz.
    pairs = []
    for i in range(0, len(temp)):
        for j in range(0, len(temp)):
            if temp[i] == temp[j]:
                pass
            else:
                pairs.append([temp[i], temp[j]])

    del count, temp, temp_path

    for i in pairs:
        # We get the min distance between the two separators
        min_val = distances_from_separator[i[0]].get(i[1])

        if i[1] in adj[i[0]]:
            # TODO: check if this if is relevant
            # An edge exists
            current_weight = adj[i[0]].get(i[1])
            current_weight = current_weight.get('weight')
            if min_val < current_weight: pass

        else:
            # An edge doesn't exist between the separators, so we create it
            # TODO: Check if this is complete. Do we have to add the weight also in the other node?
            # TODO: DO we have to update the edges? (data format)
            adj[i[0]][i[1]] = {'weight': min_val}

    del i, j, min_val, current_weight, pairs
    print()

    # [E4.1] Start iterating over {z1,...,zk}
    # [E4.2] Build range tree for zi. Construct a k-dimensional
    # range tree R of points {p(y) = (p1,...,pk)}
    # The output is stored at p_j.
    # f(p(y)) = d(z_i,y) using the monoid (Z, max)

    p_j = []
    for i in portals_of_A:
        for y in set_B:
            temp = []
            for j in distances_from_separator:
                temp.append(distances_from_separator[i].get(y) - distances_from_separator[j].get(y))

            p_j.append(temp)

        # Constructing the range tree
        root = rangeTree.initialization2(p_j)

        # [E4.3] Query range tree.
        # For each x in X (X here is the set_A), query R with l1 = ... = lk = inf
        # Output: e(x, zi; Y)
        print()

        # Calculating the ranges for the query
        r_j = []
        ntive_inf = float("-inf")
        for k in (set_A):
            for j in range(0, len(portals_of_A)):
                if j < i:
                    r_j.append(ntive_inf)
                    r_j.append(distances[k].get(i) - distances[k].get(j) - 1)
                else:
                    r_j.append(ntive_inf)
                    r_j.append(distances[k].get(i) - distances[k].get(j))


        if r_j[len(r_j)-1] == ntive_inf:
            del r_j[len(r_j)-1]

        # Query the Range Tree:
        rangeTree.dimensional_range_query(root.root, r_j[0:6])
        print()

    # [E5] Recurse on G[X]
"""
    # In rj is stored the right part of the query B for every i for evey X
    # Query box = [inf,r1] x ... x [inf,rk]
    rj = []
    temp = []
    for i in separatorList:
        for k in range(0, len(X)):
            for j in separatorList:
                if j < i:
                    temp.append(distances_X[k][i] - distances_X[k][j] - 1)
                else:
                    temp.append(distances_X[k][i] - distances_X[k][j])
            rj.append(temp)
            temp = []

    print()
    # TODO: calculate=> e(x, zi; Y)
   
"""

def main():
    test = Graphs()
    g = test.createGrid2D(3, 3)
    edges2 = []

    # Configuring edges to be readable for Dijkstra function
    for i in range(len(g)-2):
        for j in range(len(g[i]) ):
            edges = [i, g[i][j], 1]
            if edges2 == []:
                edges2 = [edges[:]]
            else:
                edges2.append(edges)
    del edges
    initial_graph = nx.grid_2d_graph(3, 3)
    # Add weight 1 in each edge of the grid
    for i in initial_graph._adj:
        for j in initial_graph._adj[i]:
            initial_graph.add_edge(i, j, weight=1)

    initial_graph = nx.convert_node_labels_to_integers(initial_graph, first_label=0, ordering='default',
                                                       label_attribute='old_labels')

    # treewidth_min_degree returns a tuple with treewidth and the corresponding decomposed tree.
    # Returns: (int, Graph) tuple
    p = approx.treewidth_min_degree(initial_graph)
    tree_decomp_graph = nx.convert_node_labels_to_integers(p[1], first_label=0, ordering='default',
                                                           label_attribute='bags')
    portals_of_A, set_A, set_B = test.skew_kseparator_tree(initial_graph, p[0], tree_decomp_graph)

    algorithm_E(initial_graph, p[0], portals_of_A, set_A, set_B)



    print()
if __name__ == "__main__":
    main()