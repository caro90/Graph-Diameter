import csv
from collections import defaultdict
from dijkstra3 import *


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

    def treeDecomposition(self, graph):
        # Returns a tree decomposition of the given graph
        print("temp")

def main():
    test = Graphs()
    g = test.createGrid2D(2, 3)
    print("-------")
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

if __name__ == "__main__":
    main()