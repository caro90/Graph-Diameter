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
        with open('points2.txt', mode='r') as csv_file:
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
    def __init__(self, dimension, point, TAssoc):

        self.nextDimNode = None
        self.point = point
        self.dimension = dimension
        self.TAssoc = TAssoc
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

    def setChildren(self, right, left):
        self.rightChild = right
        self.leftChild = left

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
        # associateT is a BST object reference to the associate tree of the current BST
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

    def printLeaves(self, v):
        if v:
            if v.leftChild is None and v.rightChild is None:
                # First print the data of node only if it is a leaf
                print(v.coordinate)

            # Then recur on left child
            self.printLeaves(v.leftChild)

            # Finally recur on right child
            self.printLeaves(v.rightChild)

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
            y=x.nextDimNode

            y.setNextDimNode(Node(3, i, None))
            self.nodeXList.append(x)
            self.nodeYList.append(y)
            self.nodeZList.append(Node(3, i, None))

        # Sorting according to x, y and z coordinates
        self.nodeXList.sort(key=operator.attrgetter('coordinate'))
        self.nodeYList.sort(key=operator.attrgetter('coordinate'))
        self.nodeZList.sort(key=operator.attrgetter('coordinate'))

        # TESTING:
        x = self.build2DRangeTree(self.nodeXList, 0, root)
        root.setRoot(x)

        root.printPreorder(root.root)
        print("--------")
        root.printPostorder(root.root)
        print("--------")
        root.printInorder(root.root)

        print("-------")
        root.printLeaves(root.root)
        range1 = [-5, 200]
        print("Query:")
        L = self.oneDRangeQuery(root.root, range1)
        for i in L:
            print(i.coordinate)
        print("-----------")
        range2 = [100, 200, -11, 99]
        L2 = self.dimensionalRangeQuery(root.root, range2)
        for j in L2:
            print(j.coordinate)

    def build2DRangeTree(self, nodeList, flag, root):
        PLeftList = []
        PRightList = []
        median = int(math.ceil(len(nodeList) / 2))
        # Construction of the associate Tree treeAssoc:
        TAssoc = None
        nextDimNodeList = []
        if len(nodeList) >= 2:
            if nodeList[0].nextDimNode is not None:
                for i in nodeList:
                    nextDimNodeList.append(i.nextDimNode)
                # Sorting the list
                nextDimNodeList.sort(key=operator.attrgetter('coordinate'))
                TAssoc = self.build2DRangeTree(nextDimNodeList, flag,root)

        if len(nodeList) == 1:
            # Base case of the recursion
            mid = Node(nodeList[0].dimension, nodeList[0].point, None)
            mid.setNextDimNode(nodeList[0].nextDimNode)
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

            mid = Node(nodeList[median - 1].dimension, nodeList[median - 1].point, nodeList[median - 1].TAssoc)
            mid.setChildren(nodeList[median - 1].rightChild, nodeList[median - 1].leftChild)
            mid.setNextDimNode(nodeList[median - 1].nextDimNode)
            if len(nodeList)> 1:
                # Make the TAssoc the associate structure of mid
                mid.TAssoc = TAssoc
            del TAssoc
            vLeft = self.build2DRangeTree(PLeftList, flag, root)
            vRight = self.build2DRangeTree(PRightList, flag, root)

            # The internal nodes are used only to guide the search path
            # For the leaves we create new Node classes
            if len(PLeftList) < 2:
                x = Node(vLeft.dimension, vLeft.point, vLeft.TAssoc)
                x.setNextDimNode(vLeft.nextDimNode)
                mid.leftChild = x
            else:
                mid.leftChild = vLeft
            if len(PRightList) < 2:
                y = Node(vRight.dimension, vRight.point, vRight.TAssoc)
                y.setNextDimNode(vRight.nextDimNode)
                mid.rightChild = y
            else:
                mid.rightChild = vRight
        return mid

    def findSplitNode(self, root, xL, xR):
        # Input: A range tree T with two values xL and xR with xL <= xR
        # Output: The node v where the paths to xL and xR split, or the leaf where both paths end.
        v = root
        while (v.leftChild is not None) and (v.rightChild is not None) and\
                ((xR <= v.coordinate) | (xL > v.coordinate)):
            if xR <= v.coordinate:
                v = v.leftChild
            else:
                v = v.rightChild
        return v

    def dimensionalRangeQuery(self, root, range):
        # Input: A dimensional range tree T and a range [x : x'] * [y : y'] * ...
        # Output: All points in T that lie in that range
        reportedList = []
        vSplit = self.findSplitNode(root, range[0], range[1])

        # if vSplit is a leaf
        if (vSplit.leftChild is None) and (vSplit.rightChild is None):
            # Checking if the point stored at vSplit must be reported
            if (vSplit.coordinate >= range[0]) & (vSplit.coordinate <= range[1]):
                reportedList.append(vSplit)
        else:
            # Follow the path to x and call oneDrangeQuery on the subtrees right of the path
            v = vSplit.leftChild
            # While v is not a leaf
            while (v.leftChild is not None) and (v.rightChild is not None):
                if (range[0] <= v.coordinate):
                    temp = self.oneDRangeQuery(v.rightChild.TAssoc, [range[2], range[3]])
                    if temp is not None:
                        for i in temp:
                            reportedList.append(i)
                    else:
                        if (range[2] <= v.rightChild.nextDimNode.coordinate) & \
                                (range[3] >= v.rightChild.nextDimNode.coordinate):
                            reportedList.append(v.rightChild.nextDimNode)
                    v = v.leftChild
                else:
                    v = v.rightChild
            # Check if the point stored at v must be reported
            if (v.coordinate >= range[0]) & (v.coordinate <= range[1])\
                        &(range[2] <= v.nextDimNode.coordinate)\
                        &(range[3] >= v.nextDimNode.coordinate):
                    reportedList.append(v.nextDimNode)


            # Similarly, follow the path from the right of vSplit to x',
            # call oneDRangeQuery with the range [y:y'] on the associate structures
            # of subtrees left of the path, and check if the point stored at the leaf
            # where the paths ends must be reported
            v = vSplit.rightChild
            # While v is not a leaf
            while (v.leftChild is not None) and (v.rightChild is not None):
                if (range[1] >= v.coordinate):
                    temp = self.oneDRangeQuery(v.leftChild.TAssoc, [range[2], range[3]])
                    if temp is not None:
                        for i in temp:
                            reportedList.append(i)
                    else:
                        if (range[2] <= v.leftChild.nextDimNode.coordinate) & \
                                (range[3] >= v.leftChild.nextDimNode.coordinate):
                            reportedList.append(v.leftChild.nextDimNode)
                    v = v.rightChild
                else:
                    v = v.leftChild

            # Check if the point stored at v must be reported
            if (v.coordinate >= range[0]) & (v.coordinate <= range[1])\
                        &(range[2] <= v.nextDimNode.coordinate)\
                        &(range[3] >= v.nextDimNode.coordinate):
                reportedList.append(v.nextDimNode)
        return reportedList

    def oneDRangeQuery(self, root, range):
        # Input: A binary search tree T and a range [x : x']
        # Output: All points stored in T that lie in that range

        if root is None:
            return

        reportedList = []
        vSplit = self.findSplitNode(root,range[0], range[1])
        if (vSplit.leftChild is None) and (vSplit.rightChild is None):
            # if vSplit is a leaf
            if (vSplit.coordinate >= range[0]) & (vSplit.coordinate <= range[1]):
                # Checking if the point stored at vSplit must be reported
                reportedList.append(vSplit)
        else:
            # Follow the path to x(stored at range[0]) and report the points in subtrees right of the path
            v = vSplit.leftChild
            # While v is not a leaf:
            while (v.leftChild is not None) and (v.rightChild is not None):
                if range[0] <= v.coordinate:
                    self.reportSubtree(v.rightChild)
                    v = v.leftChild
                else:
                    v = v.rightChild
            # Check if the point stored at the leaf v must be reported:
            if (v.coordinate >= range[0]) & (v.coordinate <= range[1]):
                reportedList.append(v)

            # Similarly, follow the path to x'(stored at range[1]), report the points in subtrees left of the path,
            # and check if the point stored at the leaf where the path ends must be reported
            v = vSplit.rightChild
            # While v is not a leaf:
            while (v.leftChild is not None) and (v.rightChild is not None):
                if range[1] >= v.coordinate:
                    self.reportSubtree(v.leftChild)
                    v = v.rightChild
                else:
                    v = v.leftChild
            # Check if the point stored at the leaf v must be reported:
            if (v.coordinate >= range[0]) & (v.coordinate <= range[1]):
                reportedList.append(v)

        return reportedList

    def reportSubtree(self, v):
        if v:
            if v.leftChild is None and v.rightChild is None:
                # First print the data of node only if it is a leaf
                print(v.coordinate)
            # Then recur on left child
            self.reportSubtree(v.leftChild)
            # Finally recur on right child
            self.reportSubtree(v.rightChild)
def main():
    testRange = RangeTree()
    testRange.initialization()

if __name__ == "__main__":
    main()