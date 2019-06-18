from DataHandler import *
from binaryTrees import *
import operator
import math


class RangeTree:
    # TODO: write doc for each function and class
    def __init__(self):
        self.dimension = 1
        self.nodeXList = []
        self.nodeYList = []
        self.nodeZList = []
        self.dynamicList = []

    def initialization(self):

        # Loading points and setting up the nodes for the BSTs
        # Read the points from the file
        root = BST(None)
        x = PointHandler()
        listOfPoints = x.insertFile_XYZval("points2.txt")

        # Creating Node objects for x, y and z coordinates
        for i in listOfPoints:
            x = Node(1, i, None, None)
            x.setNextDimNode(Node(2, i, None, None))
            y=x.nextDimNode

            y.setNextDimNode(Node(3, i, None, None))
            self.nodeXList.append(x)
            self.nodeYList.append(y)
            self.nodeZList.append(Node(3, i, None, None))

        # Sorting according to x, y and z coordinates
        self.nodeXList.sort(key=operator.attrgetter('coordinate'))
        self.nodeYList.sort(key=operator.attrgetter('coordinate'))
        self.nodeZList.sort(key=operator.attrgetter('coordinate'))

        # TESTING:
        x = self.buildRangeTree(self.nodeXList, 0, root)
        root.setRoot(x)

        print("-------")
        root.printLeaves(root.root)
        print("-------")
        root.printLeaves(root.root.TAssoc)
        print("-------")
        root.printLeaves(root.root.TAssoc.TAssoc)


        range1 = [-5, 200]
        print("Query 1 D:")
        L = self.oneDRangeQuery(root.root, range1)
        for i in L:
            print(i.coordinate)

        print("-----------")
        print("Query 2D:")
        range2 = [-5, 200, 3, 8]
        L2 = self.dimensionalRangeQuery(root.root, range2)
        for j in L2:
            print(j.coordinate)

        print("-----------")
        print("Query 3D:")
        range3 = [-5, 200, 3,8, -100, 120]
        L3 = self.dimensionalRangeQuery(root.root, range3)
        for j in L3:
            print(j.coordinate)


    def initialization2(self, pi):
        root = BST(None)
        listOfPoints = []
        x = PointHandler()
        for i in pi:
            for j in range(0, len(pi)):
                point = x.insertManually2(i[j])
                listOfPoints.append(point)

        # Creating Node objects for all the coordinates
        for i in listOfPoints:
            stop = i.listOfCoordinates
            for j in range(1, len(stop)+1):
                x = Node(j, None, None, i)
                if j != len(stop):
                    x.setNextDimNode(Node(j+1, None, None, i))
                if j == 1:
                    self.nodeXList.append(x)

        # Sorting according to the first coordinate
        self.nodeXList.sort(key=operator.attrgetter('coordinate'))
        # TESTING:
        x = self.buildRangeTree(self.nodeXList, 0, root)
        root.setRoot(x)
        print("-------")
        root.printLeaves(root.root)
        print("-------")
        root.printLeaves(root.root.TAssoc)
        print("-------")
        root.printLeaves(root.root.TAssoc.TAssoc)

        range1 = [1, 2]
        print("Query:")
        L = self.oneDRangeQuery(root.root, range1)
        for i in L:
            print(i.coordinate)

        print("a")

    def buildRangeTree(self, nodeList, flag, root):
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
                TAssoc = self.buildRangeTree(nextDimNodeList, flag,root)

        if len(nodeList) == 1:
            # Base case of the recursion
            mid = Node(nodeList[0].dimension, nodeList[0].point, None, nodeList[0].dynamicPoint)
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

            mid = Node(nodeList[median - 1].dimension, nodeList[median - 1].point, nodeList[median - 1].TAssoc, nodeList[median - 1].dynamicPoint)
            mid.setChildren(nodeList[median - 1].rightChild, nodeList[median - 1].leftChild)
            mid.setNextDimNode(nodeList[median - 1].nextDimNode)
            if len(nodeList)> 1:
                # Make the TAssoc the associate structure of mid
                mid.TAssoc = TAssoc
            del TAssoc
            vLeft = self.buildRangeTree(PLeftList, flag, root)
            vRight = self.buildRangeTree(PRightList, flag, root)

            # The internal nodes are used only to guide the search path
            # For the leaves we create new Node classes
            if len(PLeftList) < 2:
                x = Node(vLeft.dimension, vLeft.point, vLeft.TAssoc, vLeft.dynamicPoint)
                x.setNextDimNode(vLeft.nextDimNode)
                mid.leftChild = x
            else:
                mid.leftChild = vLeft
            if len(PRightList) < 2:
                y = Node(vRight.dimension, vRight.point, vRight.TAssoc, vRight.dynamicPoint)
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
        counter = 0
        while counter < len(range)-2:
            vSplit = self.findSplitNode(root, range[counter], range[counter+1])
            root = vSplit.TAssoc
            counter +=2
        #counter -=2

        # if vSplit is a leaf
        if (vSplit.leftChild is None) and (vSplit.rightChild is None):
            # Checking if the point stored at vSplit must be reported
            if (vSplit.coordinate >= range[counter-2]) & (vSplit.coordinate <= range[counter-1]):
                reportedList.append(vSplit)
        else:
            # Follow the path to x and call oneDrangeQuery on the subtrees right of the path
            v = vSplit.leftChild
            # While v is not a leaf
            while (v.leftChild is not None) and (v.rightChild is not None):
                if range[counter-2] <= v.coordinate:
                    temp = self.oneDRangeQuery(v.rightChild.TAssoc, [range[counter], range[counter+1]])
                    if temp is not None:
                        for i in temp:
                            reportedList.append(i)
                    else:
                        if (range[counter] <= v.rightChild.nextDimNode.coordinate) & \
                                (range[counter+1] >= v.rightChild.nextDimNode.coordinate):
                            reportedList.append(v.rightChild.nextDimNode)
                    v = v.leftChild
                else:
                    v = v.rightChild
            # Check if the point stored at v must be reported
            if (v.coordinate >= range[counter-2]) & (v.coordinate <= range[counter-1])\
                         &(range[counter] <= v.nextDimNode.coordinate)\
                         &(range[counter+1] >= v.nextDimNode.coordinate):
                     reportedList.append(v.nextDimNode)




            # Similarly, follow the path from the right of vSplit to x',
            # call oneDRangeQuery with the range [y:y'] on the associate structures
            # of subtrees left of the path, and check if the point stored at the leaf
            # where the paths ends must be reported
            v = vSplit.rightChild
            # While v is not a leaf


            while (v.leftChild is not None) and (v.rightChild is not None):
                if (range[counter-1] >= v.coordinate):
                    temp = self.oneDRangeQuery(v.leftChild.TAssoc, [range[counter], range[counter+1]])
                    if temp is not None:
                         for i in temp:
                            reportedList.append(i)
                    else:
                        if (range[counter] <= v.leftChild.nextDimNode.coordinate) & \
                                   (range[counter+1] >= v.leftChild.nextDimNode.coordinate):
                            reportedList.append(v.leftChild.nextDimNode)
                    v = v.rightChild
                else:
                    v = v.leftChild

                # Check if the point stored at v must be reported
            if (v.coordinate >= range[counter-2]) & (v.coordinate <= range[counter-1])\
                        &(range[counter] <= v.nextDimNode.coordinate)\
                         &(range[counter+1] >= v.nextDimNode.coordinate):
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