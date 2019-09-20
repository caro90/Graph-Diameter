import csv
from collections import defaultdict
from dijkstra3 import *
import networkx as nx
from networkx.algorithms import approximation as approx
import matplotlib.pyplot as plt
import operator



class Point:
    """

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
    """
    def insertManually2(self, listOfCoordinates):
        """
        """
        point = Point(listOfCoordinates)
        return point

    def insertFile_XYZval(self, file):
        """
        Imports the points from a csv file and creates object points

        """
        listOfPoints = []
        tempList = []
        listOfElements = []
        with open(file, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    print(f'Column names are:{", ".join(row)}')
                print(f'\t\t\t\t { row["x"]}  {row["y"]}  {row["z"]} { row["val"]}')

                for i in row:
                    listOfElements.append(row[i])

                tempList.append(listOfElements)
                listOfElements = []
                tempPoint = Point(tempList)
                listOfPoints.append(tempPoint)
                line_count += 1
        return listOfPoints


class Graphs:
    # TODO: check if it is needed and delete createGrid2D
    def createGrid2D(self, x, y):
        # Creates a grid x*y where all the weights are 1

        # Every point in the grid and its adjacent points are saved in a list
        # It creates a list of lists where in every sublist are stored the adjacent
        # points of the current point
        g = defaultdict(list)
        counter =0;
        for i in range(x):
            for j in range(y):
                # Change line
                if i + 1 <= x - 1: g[i + j + counter].append((i + j + counter + y))
                # Change column
                if j + 1 <= y - 1: g[i + j + counter].append((i + j + 1 + counter))
                # Change line
                if i - 1 >= 0: g[i + j + counter].append((i + j + counter - y))
                # Change column
                if j - 1 >= 0: g[i + j + counter].append((i + j + counter -1))
            counter = counter + (y-1)
        # Storing at the last 2 elements of the grid its x and y size
        g[len(g)].append(x)
        g[len(g)].append(y)
        return g

    # TODO: keep one out of two data_reform and document
    def data_reform(self, num_of_nodes, tw, tree_decomp_graph):
        """
        Transforms the input graph from networkx into a readable form for skew_kseparator_tree
        This function breaks the dependency of skew_kseparator_tree on networkx graph structure.

        Parameters:
        ----------
        initial_graph :
            The very first graph without any processing.

        tw: int
            The treewidth of the current tree decomposition.

        tree_decomp_graph :
            The resulting graph after applying a tree decomposition algorithm on initial_graph.

        Retuns:
        -------
        node_list: list of lists
            For each list/node or bag contains a list of vertices which came from the tree decomposition
        adjacency_list: list of lists
            For each node of the graph(the result of the tree decomposition) it keeps a list of the adjacent nodes
        node_list_set: list of sets
            The very same as node_list but in different format for more efficient calculations
        adjacency_list_set: list of sets
           ...

        """

        # Read the bags and their adjacency list and move them to a mutable object (list)
        adjacency_list = []
        node_list = []
        for x in tree_decomp_graph._adj:
            temp1 = []
            for y in tree_decomp_graph._adj[x]:
                temp1.append(y)
            adjacency_list.append(temp1)

        for x in tree_decomp_graph._node:
            node_list.append(list( tree_decomp_graph._node[x]["bags"]))

        # Creating set versions of node_list and adjacency_list
        node_list_set = [set(i) for i in node_list]
        adjacency_list_set = list()
        temp = []
        for i in range(0, len(adjacency_list)):
            for j in adjacency_list[i]:
                temp.append(j)
            adjacency_list_set.append(set(temp))
            temp = list()

        return node_list, adjacency_list, node_list_set, adjacency_list_set

    def data_reform2(self, tree_decomp_graph, set_of_nodes, initial_graph):
        """
        Transforms the input graph from networkx library into a readable form for skew_kseparator_tree
        This function breaks the dependency of skew_kseparator_tree on networkx graph structure.

        Parameters:
        ----------
        initial_graph :
            The very first graph without any processing.
        Retuns:
        -------
        node_list: list of lists
            For each list/node or bag contains a list of vertices which came from the tree decomposition
        adjacency_list: list of lists
            For each node of the graph(the result of the tree decomposition) it keeps a list of the adjacent nodes

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

        return node_dict, adjacency_dict, initial_graph_node_dict, initial_graph_adjacency_dict

    # TODO: test if needed and delete
    def test(self, set_t):
        return len(set_t)

    def skew_kseparator_tree(self, num_of_nodes, tw, tree_decomp_graph, nodes, adj, set_temp,
                             initial_graph, initial_graph_nodes, initial_graph_adj):

        """
        # TODO: write summary of the algorithm
        For k>=1, given a graph G with n > k+1 vertices and treewidth at most k,
        this function computes in linear time a subset A of V(G)

        Parameters:
        ----------
        initial_graph :
            The very first graph without any processing.

        tw: int
            The treewidth of the current tree decomposition.



        Returns:
        --------
            A graph which is the skew separator tree.

        """

        # num_of_nodes is the nodes in the initial graph(before tree decomposition)


        # num_of_nodes has to be strictly bigger than the treewidth + 1 (Requirement in Lemma 3)
        if num_of_nodes > tw + 1:
            # Transformation part:
            # [Step 1] Add vertices to each bag until each bag has exactly k+1 elements
            for i in nodes:#range(0, len(nodes)):
                while len(nodes[i]) < tw + 1:
                    # Find the adjacent nodes of current node to draw elements from there
                    # in order to keep property 3 of tree decomposition
                    for j in adj[i]:
                        diff_of_sets = nodes[j].difference(nodes[i])
                        if len(diff_of_sets) != 0:
                            nodes[i].add(next(iter(diff_of_sets)))
                        if len(nodes[i]) == tw + 1:
                            break

            #del diff_of_sets, i, j

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
        else:
            print("Error: Total vertices inside the bags <= k+1")
            return

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

        del e, i, inter, sum, temp, white_nodes, set_of_numbers, nodes_already_visited, nodes_to_be_deleted, nodes_to_visit

        # Finding the edge
        x_i = min(dict, key=dict.get)
        x_j = predecessor[x_i]
        # We now know that the edge with the lowest absolute difference is between node x_i and x_j
        nodes_to_visit = []
        # A  contains the nodes/bags of the tree decomposition
        A = set()
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
        set_A = set()
        for i in A:
            [set_A.add(j) for j in nodes[i]]
        # portals_of_A: are the nodes, of the initial graph, that have some edge incident to V(G)\set_A
        #portals_of_A = nodes[x_i].intersection(nodes[x_j])
        portals_of_A = []
        for i in set_A:
            if len( initial_graph_adj[i].difference(set_A)) >0:
                portals_of_A.append(i)

        # set_B: are the nodes of V(G) equal to right part of A (or left) along with the portals of A.
        set_B = set()
        # TODO:change name in set_temp and document it
        if len(set_temp) == 0:
            [set_B.add(i) for i in range(0, num_of_nodes)]
        else:
            [set_B.add(i) for i in set_temp]
        for i in set_A:
            set_B.remove(i)
        [set_B.add(i) for i in portals_of_A]


        return portals_of_A, set_A, set_B, A


def main():

    # Testing using my own grid structure:

    test = Graphs()
    g = test.createGrid2D(3, 3)
    print("-------")
    print(g)
    edges2 = []
    # Configuring edges to be readable for Dijkstra function
    for i in range(len(g)-2):
        for j in range(len(g[i])):
            edges = [i, g[i][j], 1]
            if edges2 == []:
                edges2 = [edges[:]]
            else:
                edges2.append(edges)

    # Graph is dijkstra's class
    graph = Graph(edges2)
    path, distances = graph.dijkstra(0, 8)
    print(path)
    print(distances)

    del i, j, g, path, distances, edges, edges2, graph
    # -------------------------------

    # Testing using networkx's structure:

    initial_graph = nx.grid_2d_graph(6, 6)
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

    # print or not the Tree Decomposition
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
    nodes, adj, initial_graph_nodes, initial_graph_adj = test.data_reform2(tree_decomp_graph, set_of_nodes, initial_graph)
    num_of_nodes = len(initial_graph._node)
    portals_of_A, set_A, set_B, A = test.skew_kseparator_tree(num_of_nodes, p[0], tree_decomp_graph, nodes, adj,
                                                              [], initial_graph, initial_graph_nodes, initial_graph_adj)
    print()

if __name__ == "__main__":
    main()