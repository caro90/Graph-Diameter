from RangeTree import *
import numpy as np
import math
import time
import fast_dijkstra as fast_dijks


def data_format(adj, graph2, flag):
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
    if flag == 1:
        for i in (adj):
            for j in adj[i]:
                edges = [i, j, next(iter(adj[i][j].values()))]
                if edges2 == []:
                    edges2 = [edges[:]]
                else:
                    edges2.append(edges)

    else:
        for key, value in adj.items():
            for j in value:
                graph2.add_edge(key, j, 1)

    return edges2, adj, graph2


def algorithm_E(initial_graph, tw, portals_of_A, set_A, set_B, tree_decomp_graph, A):
    """
    # TODO: write summary of the algorithm
    Given a graph G and a skew k-separator tree, the algorithm computes the eccentricity e(v) of every
    vertex v in V(G). We write X = union(L,Z) and Y = union(R,Z)

    Parameters:
    ----------
    initial_graph:
    tw:
    portals_of_A:
    set_A:
    set_B
    tree_decomp_graph:
    A:

    Returns:
    --------
    eccentricities:

    """
    # TODO: not be dependent on networkx, change variables, eg initial_graph._node
    flag = 2
    graph2 = fast_dijks.Graph()
    rangeTree = RangeTree()
    test = Graphs()
    distances = {}

    # edges: the edges of the initial_graph in readable format for Dijkstra
    edges, adj, graph2 = data_format(initial_graph._adj, graph2, flag)
    #graph = Graph(edges)

    # [E1] Base case of the recursion:
    # if n\ln(n) < 4tw(tw+1) then find all distances using Dijkstra's Algorithm and terminate

    # n: the number of nodes in G, |V(G)|.
    n = len(initial_graph._node)
    print("First term:", (n / math.log(n, np.e)))
    print("Second term:", 4*tw*(tw+1))
    print("----------------------------")
    if (n / math.log(n, np.e)) < 4*tw*(tw+1):
        # Dijkstra function is an object of Graph class
        for i in initial_graph._node:
            # graph.dijkstra(source, destination). The path from source to destination
            # will be saved on path variable
            #temp_path, temp_distances = graph.dijkstra(i, i)
            temp_distances = fast_dijks.dijkstra(graph2, i, i)
            distances[i] = temp_distances

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
    for i in portals_of_A:
        #temp_path, temp_distances = graph.dijkstra(i, i)
        temp_distances = fast_dijks.dijkstra(graph2, i, i)
        distances_from_separator[i] = temp_distances

    # # [E3] Add shortcuts:
    # # For each pair z,z' in Z, add the edge zz' to G, weighted by d(z,z'). Remove duplicates, retain shortest
    #
    # # Calculation of all the possible pairs of the separator nodes, zz.
    # pairs = []
    # temp = list(portals_of_A)
    # for i in range(0, len(temp)):
    #     for j in range(0, len(temp)):
    #         if temp[i] == temp[j]:
    #             pass
    #         else:
    #             pairs.append([temp[i], temp[j]])
    #
    # for i in pairs:
    #     # We get the min distance between the two separators
    #     min_val = distances_from_separator[i[0]].get(i[1])
    #
    #     if i[1] in adj[i[0]]:
    #         # An edge exists
    #         current_weight = adj[i[0]].get(i[1])
    #         current_weight = current_weight.get('weight')
    #         if min_val < current_weight:
    #             adj[i[0]][i[1]] = min_val
    #     else:
    #         # An edge doesn't exist between the separators, so we create it
    #         # TODO: Check if this is complete. Do we have to add the weight also in the other node?
    #         # TODO: DO we have to update the edges? (data format)
    #         adj[i[0]][i[1]] = {'weight': min_val}
    #         adj[i[1]][i[0]] = {'weight': min_val}

    print()
    # [E4.1] Start iterating over {z1,...,zk}
    # [E4.2] Build range tree for zi. Construct a k-dimensional
    # range tree R of points {p(y) = (p1,...,pk)}
    # The output is stored at p_j.
    # f(p(y)) = d(z_i,y) using the monoid (Z, max)

    # Calculate the distances d(x,zi) for every x in X
    # ----------------
    # Dijkstra function is an object of Graph class

    adj_x = dict()
    for i2 in set_A:
        adj_x[i2] = adj[i2]

    graph3 = fast_dijks.Graph()
    edges_x, adj_x, graph3 = data_format(adj_x, graph3, 2)
    # # graph2 is reduced only on X part of the graph
    #graph2 = Graph(edges_x)

    distances_x_zi = {}
    for k in set_A:
        # graph.dijkstra(source, destination). The path from source to destination will be saved on path variable
        temp_distances = fast_dijks.dijkstra(graph3, k, k)
        distances_x_zi[k] = temp_distances
    # ---------------------
    e_x_z = dict()
    p_y = []
    for i in range(0, len(portals_of_A)):
        for y in set_B:
            temp = []
            temp2 = []
            for j in distances_from_separator:
                temp.append(distances_from_separator[portals_of_A[i]].get(y) - distances_from_separator[j].get(y))
            # f(p(y)) = d(z_i,y) using the monoid (Z, max)
            # Finding d(z_i,y): for all the z_i in portal_of_A until the i, we select the max

            # TODO: decide which one to use as a monoid (Z, max)
            # for i2 in portals_of_A:
            #     temp2.append(distances_from_separator[i2].get(y))
            # temp.append(max(temp2))
            #------------
            # for i2 in portals_of_A:
            #     temp2.append(distances_from_separator[i2].get(y))
            #     if i2 == portals_of_A[i]:
            #         break
            # temp.append(max(temp2))
            #-----------
            temp.append(distances_from_separator[portals_of_A[i]].get(y))
            p_y.append(temp)

        # Constructing the range tree
        root = rangeTree.initialization2(p_y)
        p_y = []
        # [E4.3] Query range tree.
        # For each x in X (X here is the set_A), query R with l1 = ... = lk = inf
        # Output: e(x, zi; Y)

        # Calculating the ranges for the query
        # In rj is stored the right part of the query B for every i for every X
        # Query box = [inf,r1] x ... x [inf,rk]
        r_j = []
        ntive_inf = float("-inf")
        ptive_inf = float("inf")
        # infinity: Define a constant variable to represent infinity in an integer form
        infinity = -10000000
        query_results = {}

        for k in set_A:
            for j in range(0, len(portals_of_A)):
                if j < i:
                    r_j.append(ntive_inf)
                    r_j.append( distances_x_zi[k].get(portals_of_A[j]) - distances_x_zi[k].get(portals_of_A[i]))
                elif j >= i:
                    r_j.append(ntive_inf)
                    r_j.append( distances_x_zi[k].get(portals_of_A[j]) - distances_x_zi[k].get(portals_of_A[i]))
                else:
                    r_j.append(ntive_inf)
                    r_j.append(ptive_inf)
            # Query the Range Tree:
            # query_results: e(x, zi; Y)
            temp_query = rangeTree.dimensional_range_query(root.root, r_j, 0)
            if temp_query is not None:
                if len(temp_query) == 0:
                    query_results[k] = [infinity]
                else:
                    query_results[k] = temp_query
            else:
                temp_query = 0
            r_j = []
        # e_x_z: is e(x,zi:Y)
        # The 1st key of the dictionary is the zi, and the inner key is the x
        e_x_z[portals_of_A[i]] = query_results

    for key, value in e_x_z.items():
        for i in value:
            value[i] = max(value[i])

    # H = initial_graph.__class__()
    # H.add_nodes_from(initial_graph)
    # H.add_edges_from(initial_graph.edges)
    # [E5] Recurse on G[X]
    x = set_B.difference(set_A)
    H = initial_graph.copy()
    for j in x:
        del H._node[j]
        del H._adj[j]

    # Deleting old nodes(of initial_graph) from the adjacency list of H
    list_to_remove = []
    temp_adj = H._adj.copy()
    for i in portals_of_A:
        for j in temp_adj[i]:
            if j in x:
                list_to_remove.append(j)
        for k in list_to_remove:
             H._adj[i].pop(k)
        list_to_remove = []

    nodes, adj2, initial_graph_nodes, initial_graph_adj = test.data_reform(tree_decomp_graph, A, H)
    portals_of_A2, set_A2, set_B2, A2, B2 = test.skew_kseparator_tree(len(set_A), tw, nodes,
                                                               adj2, set_A, initial_graph_adj)


    # TODO: trick to take always the left. Doesn't work for all the lengths
    if next(iter(set_B2)) < next(iter(set_A2)):
        temp = set_B2
        set_B2 = set_A2
        set_A2 = temp
        temp2 = B2
        B2 = A2
        A2 = temp2

    # e_x_Χ: in this variable are stored the values of e(x:X)
    # e_x_Y: in this variable are stored the values of e(x:Y)
    # e_x: e(x) = max( (e;X), e(x;Y) )
    # e_z:
    e_x_Y = {}
    e_z = {}
    e_x = {}

    e_x_X = algorithm_E(H, tw, portals_of_A2, set_A2, set_B2, tree_decomp_graph, A2)
    for i in set_A:
        temp_max = 0
        for j in portals_of_A:
            if e_x_z[j].get(i) is not None:
                if temp_max < (distances_x_zi[i].get(j) + e_x_z[j].get(i)):
                    temp_max = distances_x_zi[i].get(j) + e_x_z[j].get(i)
        e_x_Y[i] = temp_max

    for i in e_x_X:
        e_x[i] = max(e_x_X[i], e_x_Y[i])

    # For the calculation of the diameter it can be completely skipped:
    # for key, value in distances_from_separator.items():
    #     e_z[key] = max(value.values())
    return e_x


def testing(initial_graph):
    test = Graphs()
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

    # -----------print or not the tree decomposition graph-----------------
    flag = 0
    if flag == 1:
        # print the adjacency list
        #   for line in nx.generate_adjlist(p[1]):
        #       print(line)
        # write edgelist to grid.edgelist
        nx.write_edgelist(tree_decomp_graph, path="grid.edgelist", delimiter=":")
        # read edgelist from grid.edgelist
        H = nx.read_edgelist(path="grid.edgelist", delimiter=":")

        nx.draw(H, with_labels=True)
        plt.show()
    del flag
    # ------------------------------------------------
    set_of_nodes = [i for i in range(0, len(tree_decomp_graph._node))]


    print("treewidth is:", p[0])
    nodes, adj, initial_graph_nodes, initial_graph_adj = test.data_reform(tree_decomp_graph, set_of_nodes, initial_graph)
    num_of_nodes = len(initial_graph._node)
    portals_of_A, set_A, set_B, A, B = test.skew_kseparator_tree(num_of_nodes, p[0], nodes, adj, [], initial_graph_adj)

    return initial_graph, portals_of_A, set_A, set_B, A, tree_decomp_graph, p, B

def main():
    # --------Importing graphs--------
    # test2 = Graphs()
    # edges = test2.import_graphs()

    # G = test2.create_graph(edges)


    # initial_graph = nx.ladder_graph(3, 90)
    G = nx.grid_2d_graph(4, 90)
    start = time.time()
    print("Start timing")
    print("------------")

    initial_graph, portals_of_A, set_A, set_B, A, tree_decomp_graph, p, B = testing(G)
    result1 = algorithm_E(initial_graph, p[0], portals_of_A, set_A, set_B, tree_decomp_graph, A)

    # [E6] Flip:
    flip = 0
    if flip == 1:
        print("Flip")
        initial_graph, portals_of_A, set_A, set_B, A, tree_decomp_graph, p, B = testing(G)
        result2 = algorithm_E(initial_graph, p[0], portals_of_A, set_B, set_A, tree_decomp_graph, B)
        maximum1 = max(result1, key=result1.get)
        maximum2 = max(result2, key=result2.get)
        print("Diameter is:", max(result1[maximum1], result2[maximum2]) )
    else:
        maximum1 = max(result1, key=result1.get)
        print("Diameter is:", result1[maximum1])

    end = time.time()
    print((end - start)/60,"min")
    print()

if __name__ == "__main__":
    main()