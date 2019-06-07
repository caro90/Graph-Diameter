import csv
import operator
import math
import random
import collections

class Point:
    def __init__(self, x, y, z, val):
        self.x = int(x)
        self.y = int(y)
        self.z = int(z)
        self.val = int(val)


class PointHandler:
    def insertManually(self, x, y, z, val):
        pointList = []
        node = Point(x, y, z, val)
        print("Insert point ")
        return node

    def insertFile(self):
        # Imports the points from a csv file and creates object points
        xlist = []
        ylist = []
        zlist = []
        listOfPoints = []
        with open('points.txt', mode='r') as csv_file:
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


class Node:
    def __init__(self, dimension, point, Tassoc):

        self.nextDimNode = None
        self.point = point
        self.dimension = dimension
        self.Tassoc = Tassoc
        # val is not strictly defined yet
        if self.dimension == 1: self.coordinate = int(point.x)
        if self.dimension == 2: self.coordinate = int(point.y)
        if self.dimension == 3:
            # if dimension is the last one we also store the value of the point
            self.coordinate = int(point.z)
            self.val = int(point.val)
        self.leftChild = None
        self.rightChild = None

    def get(self):
        return self.val

    def set(self, val):
        self.val = val

    def setNextDimNode(self, node):
        self.nextDimNode = node

    def getChildren(self):
        children = []
        if (self.leftChild != None):
               children.append(self.leftChild)
        if (self.rightChild != None):
            children.append(self.rightChild)
        return children

class BST:
    def __init__(self, root):
        self.root = root
        self.associateT = None

    def setRoot(self, root):
        self.root = root
    def setAssociateT(self, associateT):
        self.associateT = associateT

    def getRoot(self):
        return self.root

    # A function to do inorder tree traversal
    def printInorder(self, root):
        if root:
            # First recur on left child
            self.printInorder(root.leftChild)

            # then print the data of node
            print(root.coordinate),

            # now recur on right child
            self.printInorder(root.rightChild)

            # A function to do postorder tree traversal

    def printPostorder(self, root):
        if root:
            # First recur on left child
            self.printPostorder(root.leftChild)

            # the recur on right child
            self.printPostorder(root.rightChild)

            # now print the data of node
            print(root.coordinate)

            # A function to do preorder tree traversal

    def printPreorder(self, root):
        if root:
            # First print the data of node
            print(root.coordinate),

            # Then recur on left child
            self.printPreorder(root.leftChild)

            # Finally recur on right child
            self.printPreorder(root.rightChild)


class RangeTree:
    # TODO: write doc for each function and class
    def __init__(self):
        self.dimension = 1
        self.nodeXList = []
        self.nodeYList = []
        self.nodeZList = []

    def initialization(self):
        # Loading points and setting up the nodes for the BSTs
        # Read the points from the file
        root = BST(None)
        x = PointHandler()
        listOfPoints = x.insertFile()

        # Creating Node objects for x, y and z coordinates
        for i in listOfPoints:
            x = Node(1, i, None)
            x.setNextDimNode(Node(2, i, None))
            y = Node(2, i, None)
            y.setNextDimNode(Node(3, i, None))
            self.nodeXList.append(x)
            self.nodeYList.append(y)
            self.nodeZList.append(Node(3, i, None))

        # Sorting according to x, y and z coordinates
        self.nodeXList.sort(key=operator.attrgetter('coordinate'))
        self.nodeYList.sort(key=operator.attrgetter('coordinate'))
        self.nodeZList.sort(key=operator.attrgetter('coordinate'))

        self.build2DRangeTree(self.nodeXList, 0, root)

        root.printPreorder(root.root)
        print("--------")
        root.printPostorder(root.root)
        print("--------")
        root.printInorder(root.root)

    def build2DRangeTree(self, nodeList, flag, root):
        # Construction of the associate Tree treeAssoc:
        nextDimNodeList = []
        if len(nodeList >= 2):
            for i in nodeList:
                nextDimNodeList.append(i.nextDimNode)

        PLeftList = []
        PRightList = []
        median = int(math.ceil(len(nodeList) / 2))
        if len(nodeList) == 1:
            # Base case of the recursion
            mid = nodeList[0]
        else:
            if flag == 0:
                median = int(math.ceil(len(self.nodeXList) / 2))
                root.setRoot(self.nodeXList[median-1])
                # The flag is used to recognise the very first recursion and set the root node
                flag +=1
            # Create two new subsets of points: <=median and >median
            count = 1
            for i in nodeList:
                if count <= median:
                    PLeftList.append(i)
                else:
                    PRightList.append(i)
                count += 1
            vLeft = self.build2DRangeTree(PLeftList, flag, root)
            vRight = self.build2DRangeTree(PRightList, flag, root)
            mid = nodeList[median-1]
            # The internal nodes are used only to guide the search path
            # For the leaves we create new Node classes
            if len(PLeftList) < 2:
                x = Node(vLeft.dimension, vLeft.point, vLeft.Tassoc)
                mid.leftChild = x
            else:
                mid.leftChild = vLeft
            if len(PRightList) < 2:
                y = Node(vRight.dimension, vRight.point, vRight.Tassoc)
                mid.rightChild = y
            else:
                mid.rightChild = vRight

            # Make the Tassoc the associate structure of mid

        return mid


def main():
    testRange = RangeTree()
    testRange.initialization()

if __name__ == "__main__":
    main()