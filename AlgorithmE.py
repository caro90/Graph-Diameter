from RangeTree import *
import numpy as np
import math
import time
import fast_dijkstra as fast_dijks
import sys
import csv
import gc


class Distances:
    skew_separator_time = 0

    def data_format(self,adj, graph2, flag):
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


    def algorithm_E_sup(self, set_A, set_B, portals_of_A, adj, distances_from_separator, rangeTree, initial_graph, tw, tree_decomp_graph, test, A, B):
        """
        This function contains the steps E4.1 - E5 of algorithmE from the following paper:
        Multivariate Analysis Of Orthogonal Range Searching and Graph Distances Parameterized by Treewidth

        Parameters
        ----------
        set_A: list
             Contains the nodes of V(G) to the left of the portals_of_A
        set_B: list
            Contains the nodes of V(G) to the right of the portals_of_A
        portals_of_A: list
            Contains the nodes of V(G) which separate the tree decomposition and initial graph
        adj: list
            The adjacency list of the nodes in initial_graph
        distances_from_separator: list
            For each separator node the distance from all other nodes is stored.
        rangeTree: RangeTree object
        initial_graph: graph object of networkx library
        tw: int
            The treewidth of the current tree decomposition.
        tree_decomp_graph: graph object of networkx library
            tree_decomp_graph is the output of approx.treewidth_min_degree function
        test: Graphs object
        A: list
            Contains the left part of the tree decomposition nodes

        Returns
        -------
        eccentricities: list
            For each node in the graph is stored the maximum shortest path.
        """

        # [E4.1] Start iterating over {z1,...,zk}
        # [E4.2] Build range tree for zi. Construct a k-dimensional
        # range tree R of points {p(y) = (p1,...,pk)}
        # The output is stored at p_j.
        # f(p(y)) = d(z_i,y) using the monoid (Z, max)

        # Calculate the distances d(x,zi) for every x in X
        # Dijkstra function is an object of Graph class

        graph3 = fast_dijks.Graph()
        edges_x, adj_x, graph3 = self.data_format(adj, graph3, 2)

        distances_x_zi = {}
        for k in set_A:
            # graph.dijkstra(source, destination). The path from source to destination will be saved on path variable
            temp_distances = fast_dijks.dijkstra(graph3, k, k)
            distances_x_zi[k] = temp_distances
        e_x_z = dict()
        p_y = []

        for i in range(0, len(portals_of_A)):
            for y in set_B:
                temp = []
                temp2 = []
                for j in distances_from_separator:
                    if (distances_from_separator[portals_of_A[i]].get(y) is None) or (distances_from_separator[j].get(y) is None):
                        temp.append(0)
                    else:
                        temp.append(distances_from_separator[portals_of_A[i]].get(y) - distances_from_separator[j].get(y))
                # f(p(y)) = d(z_i,y) using the monoid (Z, max)
                # Finding d(z_i,y): for all the z_i in portal_of_A until the i, we select the max
                if distances_from_separator[portals_of_A[i]].get(y) is None:
                    temp.append(0)
                else:
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
                        if (distances_x_zi[k].get(portals_of_A[j] )is None) or (distances_x_zi[k].get(portals_of_A[i]) is None):
                            r_j.append(ntive_inf)
                        else:
                            r_j.append( distances_x_zi[k].get(portals_of_A[j]) - distances_x_zi[k].get(portals_of_A[i])+1)
                    elif j >= i:
                        r_j.append(ntive_inf)
                        if (distances_x_zi[k].get(portals_of_A[j]) is None) or (distances_x_zi[k].get(portals_of_A[i]) is None):
                            r_j.append(ntive_inf)
                        else:
                            r_j.append( distances_x_zi[k].get(portals_of_A[j]) - distances_x_zi[k].get(portals_of_A[i]) )
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
            rangeTree = RangeTree()
            gc.collect()

        del rangeTree
        gc.collect()

        for key, value in e_x_z.items():
            for i in value:
                value[i] = max(value[i])

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

        start1 = time.time()
        portals_of_A2, set_A2, set_B2, A2, B2, flag_separator = test.skew_kseparator_tree(len(set_A), tw, nodes,
                                                                   adj2, set_A, initial_graph_adj, tree_decomp_graph)

        end1 = time.time()
        self.skew_separator_time = self.skew_separator_time +( (end1 - start1)/60)

        # Recurse
        e_x_X = {}
        e_x_X = self.algorithm_E(H, tw, portals_of_A2, set_A2, set_B2, tree_decomp_graph, A2, B2, flag_separator)

        # e_x_Χ: in this variable are stored the values of e(x:X)
        # e_x_Y: in this variable are stored the values of e(x:Y)
        # e_x: e(x) = max( (e;X), e(x;Y) )
        # e_z:
        e_x_Y = {}
        e_z = {}
        e_x = {}

        for i in set_A:
            temp_max = 0
            for j in portals_of_A:
                if (e_x_z[j].get(i) is not None) and (distances_x_zi[i].get(j) is not None):
                    if temp_max < (distances_x_zi[i].get(j) + e_x_z[j].get(i)):
                        temp_max = distances_x_zi[i].get(j) + e_x_z[j].get(i)
                else:
                    temp_max = ntive_inf
            e_x_Y[i] = temp_max

        for i in e_x_X:
            e_x[i] = max(e_x_X[i], e_x_Y[i])

        for key, value in distances_from_separator.items():
            e_z[key] = max(value.values())

        return e_x

    def algorithm_E(self, initial_graph, tw, portals_of_A, set_A, set_B, tree_decomp_graph, A, B, flag_separator):

        """
        Given a graph G and a skew k-separator tree, the algorithm computes the eccentricity e(v) of every
        vertex v in V(G). We write X = union(L,Z) and Y = union(R,Z)

        Parameters:
        ----------
        initial_graph: graph object of networkx library
        tw: int
            The treewidth of the current tree decomposition.
        portals_of_A: list
            Contains the nodes of V(G) which separate the tree decomposition and initial graph
        set_A: list
             Contains the nodes of V(G) to the left of the portals_of_A
        set_B: list
            Contains the nodes of V(G) to the right of the portals_of_A
        tree_decomp_graph: graph object of networkx library
            tree_decomp_graph is the output of approx.treewidth_min_degree function
        A: list
            Contains the left part of the tree decomposition nodes
        Returns:
        --------
        eccentricities: list
            For each node in the graph is stored the maximum shortest path.
        """
        flag = 2
        graph2 = fast_dijks.Graph()
        rangeTree = RangeTree()
        test = Graphs()
        distances = {}

        # edges: the edges of the initial_graph in readable format for Dijkstra
        edges, adj, graph2 = self.data_format(initial_graph._adj, graph2, flag)

        # [E1] Base case of the recursion:
        # if n\ln(n) < 4tw(tw+1) then find all distances using Dijkstra's Algorithm and terminate

        # n: the number of nodes in G, |V(G)|.
        k = len(portals_of_A)
        n = len(initial_graph._node)
        # print("First term:", (n / math.log(n, np.e)))
        # print("Second term:", 4*k*(k+1))
        # print("----------------------------")
        if n <= 0:
            left_part = -100
        else:
            left_part = n / math.log(n, np.e)

        if (left_part < 4*k*(k+1)) or (flag_separator):
            # Dijkstra function is an object of Graph class
            for i in initial_graph._node:
                # graph.dijkstra(source, destination). The path from source to destination
                # will be saved on path variable
                temp_distances = fast_dijks.dijkstra(graph2, i, i)
                distances[i] = temp_distances

            # Computing the eccentricities from distances
            eccentricities = {}
            for i in distances:
               max_value = max(distances[i].values())
               eccentricities[i] = max_value

            return eccentricities

        # [E2] Distances from separator:
        #  Compute d(z,v) for every z in Z and every v in V(G), using k applications of Dijkstra's algorithm.
        distances_from_separator = {}
        for i in portals_of_A:
            temp_distances = fast_dijks.dijkstra(graph2, i, i)
            distances_from_separator[i] = temp_distances

        # TODO: not necessary step of the algorithm
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
        #         adj[i[0]][i[1]] = {'weight': min_val}
        #         adj[i[1]][i[0]] = {'weight': min_val}

        # E[4.1] - E[5] steps

        e_x = self.algorithm_E_sup(set_A, set_B, portals_of_A, adj, distances_from_separator, rangeTree, initial_graph, tw,
                        tree_decomp_graph, test, A, B)

        # [E6] Flip:
        e_y = self.algorithm_E_sup(set_B, set_A, portals_of_A, adj, distances_from_separator, rangeTree, initial_graph, tw,
                        tree_decomp_graph, test, B, A)
        e_x.update(e_y)
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
    # Measuring the time of tree decomposition:
    start = time.time()
    tree_decomp_graph = nx.convert_node_labels_to_integers(p[1], first_label=0, ordering='default',
                                                           label_attribute='bags')
    end = time.time()
    tree_decomposition_time = (end - start) / 60

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

    nodes, adj, initial_graph_nodes, initial_graph_adj = test.data_reform(tree_decomp_graph, set_of_nodes, initial_graph)
    num_of_nodes = len(initial_graph._node)
    portals_of_A, set_A, set_B, A, B, flag_separator = test.skew_kseparator_tree(num_of_nodes, p[0], nodes, adj, [], initial_graph_adj, tree_decomp_graph)

    return initial_graph, portals_of_A, set_A, set_B, A, tree_decomp_graph, p, B,tree_decomposition_time

def main():
    # Testing Grids
    a = 2
    #sys.argv = [-2, 1000]
    [sys.argv.append(i) for i in range(300, 301, 350)]

    # --------Importing graphs--------
    test = Graphs()
    # edges = test2.import_graphs()
    # G = test2.create_graph(edges)

    file = open('test.csv', mode='w')
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['Treewidth', 'Dimension', 'Log(n)', 'First term', 'Second Term', 'n', 'Diameter', 'AlgorithmE Time', 'Tree decomp Time','Skew k-separator time'])

    for i in range(1, len(sys.argv)):
        sum_algoE = 0
        sum__treedec = 0
        test1 = Distances()
        print("grid:", sys.argv[i])
        # run the same experiment many times to smooth the process.
        for j in range(0, 1, 1):
            b = int(sys.argv[i])
            G = nx.grid_2d_graph(a, b)
            start = time.time()
            # print("Start timing")
            # print("------------")

            initial_graph, portals_of_A, set_A, set_B, A, tree_decomp_graph, p, B, tree_decomp_time = testing(G)
            print("Initial dimension is:", len(portals_of_A))
            print("log(n) = ", math.log(a*b))
            print("------------------------------------------")
            print("First term:", (a*b / math.log(a*b, np.e)))
            print("Second term:", 4 * len(portals_of_A) * (len(portals_of_A) + 1))
            print("------------------------------------------")

            eccentricities = test1.algorithm_E(initial_graph, p[0], portals_of_A, set_A, set_B, tree_decomp_graph, A, B, 0)
            maximum1 = max(eccentricities, key=eccentricities.get)
            # print("Diameter is:", eccentricities[maximum1])
            end = time.time()
            # print((end - start)/60,"min")

            sum_algoE = sum_algoE + (end - start)/60
            sum__treedec = sum__treedec + tree_decomp_time
        # average the sums:
        sum_algoE = sum_algoE/(j+1)
        sum__treedec = sum__treedec/(j+1)
        writer.writerow([p[0], len(portals_of_A), math.log(a*b, np.e), (a*b / math.log(a*b, np.e)),
                             4 * len(portals_of_A) * (len(portals_of_A) + 1), a*b, eccentricities[maximum1],
                             format(sum_algoE, '.4f'), format(sum__treedec, '.4f'), format(test1.skew_separator_time, '.4f')])

        print("self.skew_separator_time", test1.skew_separator_time)
        print("algorithm time",format(sum_algoE, '.4f'))

def main2():
    # Testing partial k-trees

    # In argv are stored the different sizes of the k-trees (number of vertices)
    [sys.argv.append(i) for i in range(300, 301, 1000)]
    # Define the clique size k of the k-partial tree and p the percent of edges to remove
    k = 2
    percen = 20
    percen = [20, 40, 60]
    test = Graphs()

    file = open('test.csv', mode='w')
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(
        ['Treewidth', 'Dimension', 'Log(n)', 'First term', 'Second Term', 'n', 'k', 'p', 'Diameter', 'AlgorithmE Time',
         'Tree decomp Time','Skew k-separator time'])

    for i2 in percen:
        for i in range(1, len(sys.argv)):
            n = sys.argv[i]
            test1 = Distances()
            sum_algoE = 0
            sum__treedec = 0
            print("size of k-tree:", n)
            G = test.create_partial_ktrees(n, k, i2)
            for j in range(0, 1, 1):  # run the same experiment many times to smooth the process.

                start = time.time()
                # print("Start timing")
                # print("------------")
                initial_graph, portals_of_A, set_A, set_B, A, tree_decomp_graph, p, B, tree_decomp_time = testing(G)
                # print("Initial dimension is:", len(portals_of_A))
                # print("log(n) = ", math.log(n))
                # print("------------------------------------------")
                # print("First term:", (n / math.log(n, np.e)))
                # print("Second term:", 4 * len(portals_of_A) * (len(portals_of_A) + 1))
                # print("------------------------------------------")

                flag_separator = 0
                eccentricities = test1.algorithm_E(initial_graph, p[0], portals_of_A, set_A, set_B, tree_decomp_graph, A, B, flag_separator)
                maximum1 = max(eccentricities, key=eccentricities.get)
                # print("Diameter is:", eccentricities[maximum1])
                end = time.time()
                # print((end - start)/60,"min")

                sum_algoE = sum_algoE + (end - start) / 60
                sum__treedec = sum__treedec + tree_decomp_time
            # average the sums:
            sum_algoE = sum_algoE / (j + 1)
            sum__treedec = sum__treedec / (j + 1)
            writer.writerow([p[0], len(portals_of_A), math.log( n, np.e), (n / math.log(n, np.e)),
                             4 * len(portals_of_A) * (len(portals_of_A) + 1), n, k, i2, eccentricities[maximum1],
                             format(sum_algoE, '.4f'), format(sum__treedec, '.4f'), format(test1.skew_separator_time, '.4f')])

            # print("self.skew_separator_time", test1.skew_separator_time)
            print("algorithm time", format(sum_algoE, '.4f'))

if __name__ == "__main__":
    main2()