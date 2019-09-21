from RangeTree import *
import numpy as np
import math
from dijkstra3 import *



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


def algorithm_E(initial_graph, tw, portals_of_A, set_A, set_B, tree_decomp_graph, A):
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
    test = Graphs()
    distances = []
    distances = {}
    # edges: the edges of the initial_graph in readable format for Dijkstra
    edges, adj = data_format(initial_graph)
    graph = Graph(edges)
    # [E1] Base case of the recursion:
    # if n\ln(n) < 4tw(tw+1) then find all distances using Dijkstra's Algorithm and terminate

    # Dijkstra function is an object of Graph class
    for i in initial_graph._node:
        # graph.dijkstra(source, destination). The path from source to destination will be saved on path variable
        temp_path, temp_distances = graph.dijkstra(i, i)
        distances[i] = temp_distances

    # n: the number of nodes in G, |V(G)|.
    n = len(initial_graph._node)
    print((n / math.log(n, np.e)))
    if (n / math.log(n, np.e)) < 4*tw*(tw+1):
        # Computing the eccentricities from distances
        eccentricities = {}
        for i in distances:
           max_value = max(distances[i].values())
           eccentricities[i] = max_value

        return eccentricities

        del temp_path, temp_distances, i, max_value, edges

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
            #TODO : Check why pass?
            if min_val < current_weight: pass

        else:
            # An edge doesn't exist between the separators, so we create it
            # TODO: Check if this is complete. Do we have to add the weight also in the other node?
            # TODO: DO we have to update the edges? (data format)
            adj[i[0]][i[1]] = {'weight': min_val}

    print()
    # [E4.1] Start iterating over {z1,...,zk}
    # [E4.2] Build range tree for zi. Construct a k-dimensional
    # range tree R of points {p(y) = (p1,...,pk)}
    # The output is stored at p_j.
    # f(p(y)) = d(z_i,y) using the monoid (Z, max)

    e_x_z = dict()
    p_y = []
    for i in portals_of_A:
        for y in set_B:
            temp = []
            for j in distances_from_separator:
                temp.append(distances_from_separator[i].get(y) - distances_from_separator[j].get(y))
            # f(p(y)) = d(z_i,y) using the monoid (Z, max)
            # Finding d(z_i,y)
            temp2 = []
            for i2 in portals_of_A:
               temp2.append(distances_from_separator[i2].get(y))
            temp.append(max(temp2))
            p_y.append(temp)

        del y, j, i2, temp, temp2
        # Constructing the range tree
        root = rangeTree.initialization2(p_y)

        # [E4.3] Query range tree.
        # For each x in X (X here is the set_A), query R with l1 = ... = lk = inf
        # Output: e(x, zi; Y)
        print()

        # Calculating the ranges for the query
        # In rj is stored the right part of the query B for every i for every X
        # Query box = [inf,r1] x ... x [inf,rk]
        r_j = []
        ntive_inf = float("-inf")
        # infinity: Define a constant variable to represent infinity in an integer form
        infinity = -10000000
        query_results = {}

        for k in set_A:
            for j in range(0, len(portals_of_A)):
                if j < i:
                    r_j.append(ntive_inf)
                    r_j.append(distances[k].get(i) - distances[k].get(j) - 1)
                else:
                    r_j.append(ntive_inf)
                    r_j.append(distances[k].get(i) - distances[k].get(j))
            # Query the Range Tree:
            # query_results: e(x, zi; Y)
            temp_query = rangeTree.dimensional_range_query(root.root, r_j, 0)
            if len(temp_query) == 0:
                query_results[k] = [infinity]
            else:
                query_results[k] = temp_query
            r_j = []
        # e_x_z: is e(x,zi:Y)
        # The 1st key of the dictionary is the zi, and the inner key is the x
        e_x_z[i] = query_results

    for key, value in e_x_z.items():
        for i in value:
            value[i] = max(value[i])

    print()
    # [E5] Recurse on G[X]

    x = set_B.difference(set_A)
    initial_graph2 = initial_graph
    for j in x:
        del initial_graph2._node[j]
        del initial_graph2._adj[j]
    # TODO: check if there is need to perform tree decomposition again!!!
    p2 = approx.treewidth_min_degree(initial_graph2)
    tree_decomp_graph2 = nx.convert_node_labels_to_integers(p2[1], first_label=0, ordering='default',
                                                           label_attribute='bags')
    tw2 = p2[0]
    temp_nodes = []
    [temp_nodes.append(i) for i in range(0, len(tree_decomp_graph2._node))]

    nodes, adj, initial_graph_nodes, initial_graph_adj = test.data_reform2(tree_decomp_graph2, temp_nodes, initial_graph2)
    portals_of_A2, set_A2, set_B2, A2 = test.skew_kseparator_tree(len(set_A), tw2, tree_decomp_graph2, nodes,
                                                               adj, set_A, initial_graph2,  initial_graph_nodes, initial_graph_adj )

    # e_x_Χ: in this variable are stored the values of e(x:X)
    # e_x_Y: in this variable are stored the values of e(x:Y)
    # e_x: e(x) = max( (e;X), e(x;Y) )
    # e_z:
    e_x_Y = {}
    e_z = {}
    e_x_Χ = {}
    e_x = {}

    e_x_X = algorithm_E(initial_graph2, tw2, portals_of_A2, set_A2, set_B2, tree_decomp_graph2, A2)
    for i in set_A:
        temp_max = 0
        for j in portals_of_A:
            if e_x_z[j].get(i) is not None:
                if temp_max < (distances[i].get(j) + e_x_z[j].get(i)):
                    temp_max = distances[i].get(j) + e_x_z[j].get(i)
            else:
                if temp_max < (distances[i].get(j)):
                    temp_max = distances[i].get(j)
        e_x_Y[i] = temp_max

    for i in e_x_X:
        # if e_x_X[i] and e_x_Y[i]:
        e_x[i] = max(e_x_X[i], e_x_Y[i])
    for key, value in distances_from_separator.items():
        e_z[key] = max(value.values())

    print()
    return e_x

def testing():
    test = Graphs()

    initial_graph = nx.ladder_graph(70)
    # initial_graph = nx.star_graph(10)
    # initial_graph = nx.grid_2d_graph(4,4)

    # print or not the Graph
    flag = 0
    if flag == 0:
        # print the adjacency list
        #   for line in nx.generate_adjlist(p[1]):
        #       print(line)
        # write edgelist to grid.edgelist
        nx.write_edgelist(initial_graph, path="grid.edgelist", delimiter=":")
        # read edgelist from grid.edgelist
        H = nx.read_edgelist(path="grid.edgelist", delimiter=":")

        nx.draw(H, with_labels=True)
        plt.show()
    del flag

    # Add weight 1 in each edge of the graph
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
    set_of_nodes = [i for i in range(0, len(tree_decomp_graph._node))]

    nodes, adj, initial_graph_nodes, initial_graph_adj = test.data_reform2(tree_decomp_graph, set_of_nodes,
                                                                           initial_graph)
    num_of_nodes = len(initial_graph._node)
    portals_of_A, set_A, set_B, A = test.skew_kseparator_tree(num_of_nodes, p[0], tree_decomp_graph, nodes, adj, [],
                                                              initial_graph, initial_graph_nodes, initial_graph_adj)

    return  initial_graph, portals_of_A, set_A, set_B, A, tree_decomp_graph, p

def main():
    initial_graph, portals_of_A, set_A, set_B, A, tree_decomp_graph, p  = testing()

    result1 = algorithm_E(initial_graph, p[0], portals_of_A, set_A, set_B, tree_decomp_graph, A)
    # [E6] Flip:
    initial_graph, portals_of_A, set_A, set_B, A, tree_decomp_graph, p = testing()
    result2 = algorithm_E(initial_graph, p[0], portals_of_A, set_B, set_A, tree_decomp_graph, A)
    print()


    print()
if __name__ == "__main__":
    main()