import csv
from collections import defaultdict
from dijkstra3 import *
import networkx as nx
from networkx.algorithms import approximation as approx
import matplotlib.pyplot as plt



class Point:
    def __init__(self, x, y, z, val):
        self.x = int(x)
        self.y = int(y)
        self.z = int(z)
        self.val = int(val)

class DynamicPoint:
    def __init__(self, listOfCoordinates):
        self.listOfCoordinates = listOfCoordinates

class PointHandler:

    def insertManually(self, x, y, z, val):
        pointList = []
        node = Point(x, y, z, val)
        print("Insert point ")
        return node

    def insertManually2(self, listOfCoordinates):
        # Handles the dynamicPoint Class
        point = DynamicPoint(listOfCoordinates)
        return point

    def insertFile_XYZval(self, file):
        # Imports the points from a csv file and creates object points
        xlist = []
        ylist = []
        zlist = []
        listOfPoints = []
        with open(file, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    print(f'Column names are:{", ".join(row)}')
                print(f'\t\t\t\t { row["x"]} { row["y"]} { row["z"]} { row["val"]}')
                xlist.append(row["x"])
                ylist.append(row["y"])
                zlist.append(row["z"])
                tempPoint = Point(row["x"], row["y"], row["z"], row["val"])
                listOfPoints.append(tempPoint)
                line_count += 1
        return listOfPoints


class Graphs:
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

    # TODO check relevance and delete function
    def gridSeparator(self, grid, k):
        # Given a grid returns the indices of a k - separator tree Zt (if it exists)
        # which separates the grid in its left and right part
        # and also returns the left and the right part

        # If the lines of the grid are more than k then no k - separator exists for this grid
        if len(grid)-1 > k & len(grid)-2 > k:
            print("No such k separator exists for the given graph")
            return None
        else:
            # Find the smallest dimension so we can have a minimum tree decomposition
            y = grid[len(grid) - 1]
            y = y[0]
            x = grid[len(grid) - 2]
            x = x[0]
            if y >= x:
                dim = y
            else:
                dim = x

            indices = []
            X = []
            Y = []
            step = dim
            dim = round(y / 2)
            stop = len(grid) - 2
            # X corresponds to the left part after the removal of the separator tree
            # Y to the right part
            for i in range(dim-1, stop-1, step):
                indices.append(i)
                for j in range(i-(dim-1), i + dim-2):
                    X.append(j)
                if y%2 == 0:
                    for k in range(i+1, i + dim+1):
                        Y.append(k)
                else:
                    for k in range(i+1, i + dim):
                        Y.append(k)
        return indices, X, Y

    def skew_kseparator_tree(self, G, tw, tree_decomp):
        # TODO: Generic form of the function not to be dependent on networkX
        # For k>=1, given a graph G with n > k+1 vertices and treewidth at most k,
        # this function computes in linear time a subset A of V(G)

        # num_of_nodes is the nodes in the initial graph(before tree decomp)
        num_of_nodes = len(G._node)

        if num_of_nodes > tw + 1:
            # Read the bags and their adjacency list and move them to a mutable object (list)
            adjacency_list = []
            node_list = []
            for x in tree_decomp._adj:
                temp1 = []
                for y in tree_decomp._adj[x]:
                    temp1.append(list(y))
                adjacency_list.append(temp1)

            for x in tree_decomp._node:
                node_list.append(list(x))

            # Creating set versions of node_list and adjacency_list
            node_list_set = [set(i) for i in node_list]
            adjacency_list_set = list()
            temp = []
            for i in range(0, len(adjacency_list)):
                for j in adjacency_list[i]:
                    temp.append(set(j))
                adjacency_list_set.append(temp)
                temp = list()

            # Transformation part:
            # [Step 1] Add vertices to each bag until each bag has exactly k+1 elements
            for i in range(0, len(node_list)):
                i_temp = i

                while len(node_list[i]) < tw + 1:
                #if len(node_list[i]) < tw + 1:
                    # Use of sets for more efficient calculations
                    current_element = set(node_list[i])
                    if i_temp < len(node_list)-1:
                        temp2 = node_list[i_temp+1]
                        temp2 = set(temp2)
                        temp3 = list(temp2.difference(current_element))
                    else:
                        temp2 = node_list[i_temp - 1]
                        temp2 = set(temp2)
                        temp3 = list(temp2.difference(current_element))

                    if len(temp3) != 0:
                        # Update the adjacency lists of the neighbors of i
                        for j in adjacency_list[i_temp]:
                            temp = set(j)
                            node_list_set = [set(k_temp) for k_temp in node_list]
                            counter = -1
                            flag = 0
                            for k in node_list_set:
                                counter += 1
                                if len(k.difference(temp)) == 0:
                                    break
                            # Update the adjacency list of the neighbor of the current element
                            for k in adjacency_list[counter]:
                                temp = set(k)
                                if len(current_element.difference(temp)) == 0:
                                    k.append((temp3[0]))

                        # Update node i
                        node_list[i].append(temp3[0])
                    if len(temp3) > 0:
                        temp3.pop(0)
                    if len(temp3) == 0:
                        i_temp = i_temp + 1
                    if i_temp >= len(node_list):
                        # TODO: Check if property 3 of tree decomposition holds
                        # (use to take elements olny from adjacent nodes G._node.index(i))
                        break
                        #i_temp = 0
            del i_temp, temp3, i, j
            # [Step 2] Contract any edge ij in E(T) whenever Xi=Xj.
            # It now holds that any two nodes i j of T are different

            nodes_to_be_deleted = []
            for i in range(0, len(node_list_set)-1):
                for j in range(i+1, len(node_list_set)):
                    if len(node_list_set[j].difference(node_list_set[i])) == 0:
                        # if 2 bags contain exactly the same elements
                        # 1st contract their adjacency lists
                        for l in adjacency_list_set[i]:
                            for p in range(0, len(adjacency_list_set[j])):
                                if l in adjacency_list_set[j][p]:
                                    adjacency_list[i].append(list(adjacency_list_set[j][p]))

                        nodes_to_be_deleted.append(j)

            nodes_to_be_deleted = (list(set(nodes_to_be_deleted)))
            nodes_to_be_deleted.sort(reverse=True)
            for j in nodes_to_be_deleted:
                # Removing duplicates

                # Delete node j since j = i
                del(node_list_set[j])
                del (node_list[j])
                del (adjacency_list_set[j])
                del (adjacency_list[j])

            # [Step 3] Finally, for each node i in T of degree at least k+2 we create k+1 new bags
            # TODO: step3 (doesn't apply on grids)

            del i, j, k, counter, flag, temp, temp1, temp2, current_element, nodes_to_be_deleted, x, y
        else:
            print("Error: Total vertices inside the bags <= k+1")
            return

        # Finding separator set A

        # Assign each vertex of G to one and only one of the bags where it appears,
        # and define the weight of a bag to be the number of vertices that were
        # assigned to it

        for i in range(0, len(node_list)):
            k = 0
            temp = len(node_list[i])
            while k < temp:
                current_element = node_list[i][k]
                # if is used for to maintain a balance on the number elements inside the bags.
                if k % 2 == 0:
                    for j in range(i+1, len(node_list)):
                        if current_element in node_list[j]:
                            # This "if" is for not letting a bag to be completely empty
                            if len(node_list[j]) == 1:
                                node_list[i].pop(node_list[i].index(current_element))
                                break
                            else:
                                node_list[j].pop(node_list[j].index(current_element))
                else:
                    flag = 0
                    for j in range(i+1, len(node_list)):
                        if current_element in node_list[j]:
                            flag = 1
                            for j2 in range(j+1, len(node_list)):
                                if current_element in node_list[j2]:

                                        node_list[j2].pop(node_list[j2].index(current_element))

                    if flag == 1:
                        # current element found somewhere else and it can be removed from the
                        # 1st bag
                        node_list[i].pop(node_list[i].index(current_element))

                temp = len(node_list[i])
                k = k + 1

        print()


def main():

    test = Graphs()
    g = test.createGrid2D(2, 3)
    print("-------")
    print(g)
    edges2 = []
    # Configuring edges to be readable for Dijkstra function
    for i in range(len(g)-2):
        for j in range(len(g[i]) ):
            edges = [i, g[i][j], 1]
            if edges2 == []:
                edges2 = [edges[:]]
            else:
                edges2.append(edges)

    separatorList, X, Y = test.gridSeparator(g, 100)
    print(separatorList)
    print("-------")
    graph = Graph(edges2)
    path, distances = graph.dijkstra(0, 1)
    print(path)
    print(distances)

    # Testing
    G = nx.grid_2d_graph(5, 5)
    p = approx.treewidth_min_degree(G)
    print(p[1])

    # print the adjacency list
    for line in nx.generate_adjlist(p[1]):
        print(line)
    # write edgelist to grid.edgelist
    nx.write_edgelist(p[1], path="grid.edgelist", delimiter=":")
    # read edgelist from grid.edgelist
    H = nx.read_edgelist(path="grid.edgelist", delimiter=":")

    nx.draw(H, with_labels=True)
    plt.show()

    test.skew_kseparator_tree(G, p[0], p[1])

if __name__ == "__main__":
    main()