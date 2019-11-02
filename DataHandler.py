import csv
from collections import defaultdict
import networkx as nx
from networkx.algorithms import approximation as approx
import matplotlib.pyplot as plt
from pathlib import Path
import os
import numpy as np


class Point:
    """
    This class is mainly used from Node class to store the initial list of points. Every point of the form (x, y,..., z)
    either imported or created is stored in one of those objects.

    Parameters
    ----------
    tempList: list
        In this list is stored all the coordinates of the points.

    Attributes
    ----------
    pointList: list of integers
        The same as tempList but casted as integers. This casting is useful when the points are drawn from files in
        which they are read as strings.
    """
    def __init__(self, tempList):
        self.pointList = []
        # Changing a list of strings into a list of integers
        temp = tempList.pop()
        for i in range(0, len(temp)):
            temp2 = temp.pop(i)
            self.pointList.append(int(temp2))
            temp.insert(i, temp2)


class PointHandler:
    """
    An auxiliary class to manipulate the Point objects.
    """
    def insertManually2(self, listOfCoordinates):
        """
        This function creates and returns Point objects. It is mainly used for points that have been created and not
        imported.

        Parameters
        ----------
        listOfCoordinates: list

        Returns
        -------
        point: Point object
        """
        point = Point(listOfCoordinates)
        return point

    def read_file(self, file):
        """
        Imports the points from a csv file and creates object points.

        Parameters
        ----------
        file:
            The name of the file to be opened and from which the points will be drawn.
        Returns
        --------
        listOfPoints: list
        """
        listOfPoints = []
        tempList = []
        listOfElements = []
        with open(file, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            line_count = 0
            for row in csv_reader:
                if line_count == 0:pass
                    #print(f'Column names are:{", ".join(row)}')
                #print(f'\t\t\t\t { row["x"]}  {row["y"]}  {row["z"]} {row["w"]} {row["u"]} { row["val"]}')

                for i in row:
                    listOfElements.append(row[i])

                tempList.append(listOfElements)
                listOfElements = []
                tempPoint = Point(tempList)
                listOfPoints.append(tempPoint)
                line_count += 1
        return listOfPoints


class Graphs:
    """
    The main class tha creates, imports and manipulates graphs

    """

    # TODO:check if it is needed this function
    def create_graph(self, edges):
        G = nx.Graph()
        G.add_nodes_from([0])
        for i in edges:
            if i[0] not in G._node:
                G.add_nodes_from([i[0]])
            G.add_edge(i[0], i[1])
        return G

    def create_partial_ktrees(self, n, k, p):
        """
        Generates partial k-trees using the parametric model: (n,k,p)
        """
        # Returns a caveman graph of l cliques of size k.
        G = nx.caveman_graph(1, k+1)
        # Insert new nodes
        for i in range(0, n-k-1):
            G.add_node(i+k+1)
            adjacent_nodes = set()

            while len(adjacent_nodes) < k:

                cliques_of_G = nx.cliques_containing_node(G)

                temp = int(np.round(np.random.uniform(low=0.0, high=len(cliques_of_G)-1, size=None)))
                temp_list = cliques_of_G.get(temp)
                temp2 = int(np.round(np.random.uniform(low=0.0, high=len(temp_list) - 1, size=None)))
                temp_list = temp_list[temp2]
                if len(temp_list) >= k:
                    while(len(temp_list) != k):
                        del temp_list[0]
                    [adjacent_nodes.add(i) for i in temp_list]
            for j in list(adjacent_nodes):
                G.add_edge(i+k+1, j)

        a = set()
        b = set()
        pairs = set()
        # Remove p percent edges from the k-tree uniformly at random
        if p > 0:
            num = G.number_of_edges()
            num = int(num * (p/100) )
            # Finding the edges to remove without replacement
            while len(pairs) < num:
                temp1 = 0
                temp2 = 0
                while temp1 == temp2:
                    temp1 = int(np.round(np.random.uniform(low=0.0, high=n, size=None)))
                    temp2 = int(np.round(np.random.uniform(low=0.0, high=n, size=None)))
                temp_pairs = set()
                temp_pairs.add(temp1)
                temp_pairs.add(temp2)
                if temp_pairs not in pairs:
                    pairs.add(frozenset(temp_pairs))

            pairs = list(pairs)
            pairs_new = []
            [pairs_new.append(list(i)) for i in pairs]
            for i in range(0, num):
                if G.has_edge(pairs_new[i][0], pairs_new[i][1]):
                    G.remove_edge(pairs_new[i][0], pairs_new[i][1])
        return G

    def import_graphs(self):
        temp1 = os.path.join('Users', 'Chris', 'PycharmProjects','AlgorithmE_eccentricities','email-Eu-core')
        data_folder = Path(temp1)
        data_folder2 = Path("email-Eu-core_network-csv")

        # Converting a space delimited to a csv
        input_file = open(data_folder, 'r')
        output_file = open(data_folder2, 'w')
        input_file.readline()  # skip first line
        for line in input_file:
            (a, b) = line.strip().split(' ')
            output_file.write(','.join([a, b]) + '\n')
        input_file.close()
        output_file.close()

        # Import the csv
        edges = []
        with open(data_folder2, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    temp_elem = []
                    for i in row:
                        temp_elem.append(int(row[i]))
                    edges.append(temp_elem)

        return edges

    def data_reform(self, tree_decomp_graph, set_of_nodes, initial_graph):
        """
        Transforms the input graph from networkx library into a readable form for skew_kseparator_tree
        This function breaks the dependency of skew_kseparator_tree on networkx graph structure.
        Parameters:
        ----------
        tree_decomp_graph: Graph object from networkx library
            The tree decomposition graph of initial_graph
        set_of_nodes: list
            This list stores the nodes to be drawn from tree_decomp_graph. This variable is mainly useful during the
            recursion where only a subset of nodes of the nodes of tree_decomp_graph are needed.
        initial_graph:
            The very first graph without any processing.
        Retuns:
        -------
        node_dict: dictionary
            The bags of the tree decomposition graph along with its initial V(G) nodes.
        adjacency_dict: dictionary
            For each node of the tree decomposition graph it keeps a list of the adjacent nodes
        initial_graph_node_dict: dictionary
        initial_graph_adjacency_dict: dictionary
        """
        node_dict = {}
        adjacency_dict = {}
        initial_graph_node_dict = {}
        initial_graph_adjacency_dict = {}
        temp_nodes = set()
        [temp_nodes.add(i) for i in initial_graph]
        for i in initial_graph:
            initial_graph_node_dict[i] = set([i])
            initial_graph_adjacency_dict[i] = set(initial_graph._adj[i])
            extra_elem = initial_graph_adjacency_dict[i].difference(temp_nodes)
            if len( extra_elem) > 0:
                for e in extra_elem:
                    initial_graph_adjacency_dict[i].remove(e)

        for i in set_of_nodes:
            node_dict[i] = set(tree_decomp_graph._node[i]["bags"])
            adjacency_dict[i] = set(tree_decomp_graph._adj[i])
            extra_elem = adjacency_dict[i].difference(set_of_nodes)
            if len(extra_elem) > 0:
                for e in extra_elem:
                    adjacency_dict[i].remove(e)

        return node_dict, adjacency_dict, initial_graph_node_dict, initial_graph_adjacency_dict

    def skew_kseparator_tree(self, num_of_nodes, tw, nodes, adj, set_temp, initial_graph_adj):
        """
        For k>=1, given a graph G with n > k+1 vertices and treewidth at most k,
        this function computes in linear time a subset A of V(G)

        Parameters:
        ----------
        num_of_nodes: int
            The number of nodes in the initial graph(before tree decomposition
        tw: int
            The treewidth of the current tree decomposition.
        nodes:
        adj:
        set_temp:
        initial_graph_adj:

        Returns:
        --------
        portals_of_A:
        set_A:
        set_B:
        A:
        B:

        """
        A = set()
        set_B = set()
        set_A = set()
        # portals_of_A: are the nodes, of the initial graph, that have some edge incident to V(G)\set_A
        portals_of_A = []
        # num_of_nodes has to be strictly bigger than the treewidth + 1 (Requirement in Lemma 3)
        if num_of_nodes > tw + 1:
            # Transformation part:
            # [Step 1] Add vertices to each bag until each bag has exactly k+1 elements
            for i in nodes:
                while len(nodes[i]) < tw + 1:
                    # Find the adjacent nodes of current node to draw elements from there
                    # in order to keep property 3 of tree decomposition
                    for j in adj[i]:
                        diff_of_sets = nodes[j].difference(nodes[i])
                        if len(diff_of_sets) != 0:
                            nodes[i].add(next(iter(diff_of_sets)))
                        if len(nodes[i]) == tw + 1:
                            break

            # [Step 2] Contract any edge ij in E(T) whenever Xi == Xj.
            # It now holds that any two nodes i j of T are different
            nodes_to_be_deleted = []
            for i in nodes:
                # TODO: delete this line if there is no need to start from the children
                # Checking if the current bag contains exactly the same elements with one of its neighbors
                for e in adj[i].copy():
                    if len(nodes[e].difference(nodes[i])) == 0:
                        # Removing node -e from the adjacency list of the other nodes
                        # -i is the node that will be kept, -e the one that will be removed
                        adj[i].remove(e)
                        for j in adj:
                            if e in adj[j]:
                                adj[j].remove(e)
                                # Updating current node's adjacency list to point to node -i that will be kept
                                # Updating also node's -i adjacency list
                                adj[j].update([i])
                                adj[i].update([j])
                        # Permanently deleting node -e
                        nodes_to_be_deleted.append(e)
                        adj[e].clear()
            for j in nodes_to_be_deleted:
                 del nodes[j], adj[j]

            # [Step 3] For each node i in T of degree at least k+2 we create k+1 new bags
            # TODO: Doesn't apply on grids. Fill it later

            # [Step 4] Finding separator set A
            # Assign each vertex of G to one and only one of the bags where it appears,
            # and define the weight of a bag to be the number of vertices that were
            # assigned to it

            set_of_numbers = set()
            if len(set_temp) == 0:
                [set_of_numbers.add(i) for i in range(0, num_of_nodes)]
            else:
                [set_of_numbers.add(i) for i in set_temp]

            weights = {}
            for i in nodes:
                inter = nodes[i].intersection(set_of_numbers)
                if len(inter) != 0:
                    set_of_numbers = set_of_numbers.difference(inter)
                    weights[i] = len(inter)
                 # TODO: check if one bag can have 0 elements
                else:
                    weights[i] = 0

            # Finding the edge that minimizes the absolute value of the difference
            # between the weights of the two trees T-ij

            # Implementing a non recursive DFS
            nodes_to_visit = []
            nodes_already_visited = set()
            # white_nodes: nodes that still have children to be examined
            white_nodes = []
            my_keys = list(nodes.keys())
            nodes_to_visit.append(my_keys[0])
            dict = {}
            dict_sum = {}
            for i in nodes:
                dict_sum[i] = 0
            predecessor = {}
            while len(nodes_to_visit) > 0:
                current_node = nodes_to_visit.pop()
                white_nodes.append(current_node)
                nodes_already_visited.update([current_node])

                # Checking if current node is not leaf
                if len(adj[current_node].difference(nodes_already_visited)) > 0:
                    for i in adj[current_node]:
                        if i not in nodes_already_visited:
                            nodes_to_visit.append(i)
                            predecessor[i] = current_node
                else:
                    # Current node is a leaf
                    dict_sum[current_node] = weights[current_node]
                    dict[current_node] = abs(dict_sum[current_node] - abs(num_of_nodes - dict_sum[current_node]))
                    white_nodes.pop()

                while len(white_nodes) > 0 and len(nodes_to_visit) == 0:
                    temp = white_nodes.pop()
                    sum = 0
                    for i in adj[temp]:
                        sum = dict_sum[i] + sum
                    dict_sum[temp] = weights[temp] + sum
                    dict[temp] = abs(dict_sum[temp] - abs(num_of_nodes - dict_sum[temp]))

            #del e, i, inter, sum, temp, white_nodes, set_of_numbers, nodes_already_visited, nodes_to_be_deleted, nodes_to_visit

            # Finding the edge
            nodes_to_visit = []
            # A  contains the nodes/bags of the tree decomposition

            x_i = min(dict, key=dict.get)
            if len(predecessor) > 0: # TODO:check this change
                x_j = predecessor[x_i]
                # We now know that the edge with the lowest absolute difference is between node x_i and x_j
                nodes_to_visit.append(x_i)
                while len(nodes_to_visit) > 0:
                    current_node = nodes_to_visit.pop()
                    A.add(current_node)
                    for i in adj[current_node]:
                        # i != x_j : in order to force DFS to give us only the children of x_i
                        if i not in A and i != x_j:
                            nodes_to_visit.append(i)
            # set_A is a subset of V(G)
            # set_A contains the nodes, of the initial graph, that were inside the node/bags A.

            for i in A:
                [set_A.add(j) for j in nodes[i]]

            for i in set_A:
                if len( initial_graph_adj[i].difference(set_A)) > 0:
                    portals_of_A.append(i)

            # Creating B:
            temp_node_keys = set( nodes.keys())
            B = temp_node_keys.difference(A)

            # set_B: are the nodes of V(G) equal to right part of A (or left) along with the portals of A.

            # TODO:change name in set_temp and document it
            if len(set_temp) == 0:
                [set_B.add(i) for i in range(0, num_of_nodes)]
            else:
                [set_B.add(i) for i in set_temp]
            for i in set_A:
                set_B.remove(i)
            [set_B.add(i) for i in portals_of_A]

            return portals_of_A, set_A, set_B, A, B, 0
        else:
            # Base case
            print("Total vertices inside the bags <= k+1")
            B = 0
            flag_separator = 1
            return portals_of_A, set_A, set_B, A, B, flag_separator


def main():
    # Testing using networkx's structure:
    test = Graphs()

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
    # Print or not the Tree Decomposition
    flag = 1
    if flag == 0:
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

    set_of_nodes = [i for i in range(0, len(tree_decomp_graph._node))]
    nodes, adj, initial_graph_nodes, initial_graph_adj = test.data_reform(tree_decomp_graph, set_of_nodes, initial_graph)
    num_of_nodes = len(initial_graph._node)
    portals_of_A, set_A, set_B, A, B = test.skew_kseparator_tree(num_of_nodes, p[0], nodes, adj,
                                                              [], initial_graph_adj)
    print()

if __name__ == "__main__":
    main()