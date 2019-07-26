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
        list_of_points = x.insertFile_XYZval("points2.txt")

        # Creating Node objects for x, y and z coordinates
        for i in list_of_points:
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
        x = self.build_range_tree(self.nodeXList, 0, root)
        root.setRoot(x)

        print("---Post Order----")
        root.printPostorder(root.root)
        print("----1st Dim---")
        root.printLeaves(root.root)
        print("---2nd Dim----")
        root.printLeaves(root.root.TAssoc)
        print("----3rd Dim---")
        root.printLeaves(root.root.TAssoc.TAssoc)
        print("-------")

        range1 = [-5, 200]
        print("Query 1 D:")
        L = self.one_d_range_query(root.root, range1)
        for i in L:
            print(i.coordinate)

        print("-----------")
        print("Query 2D:")
        range2 = [-5, 200, 3, 8]
        L2 = self.dimensional_range_query(root.root, range2)
        for j in L2:
            print(j.coordinate)

        print("-----------")
        print("Query 3D:")
        range3 = [-5, 200, -100, 100, -100, 120]
        L3 = self.dimensional_range_query(root.root, range3)
        for j in L3:
            print(j.coordinate)

    def initialization2(self, pi):
        root = BST(None)
        list_of_points = []
        x = PointHandler()
        for i in pi:
            for j in range(0, len(pi)):
                point = x.insertManually2(i[j])
                list_of_points.append(point)

        # Creating Node objects for all the coordinates
        for i in list_of_points:
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
        x = self.build_range_tree(self.nodeXList, 0, root)
        root.setRoot(x)
        print("-------")
        root.printLeaves(root.root)
        print("-------")
        root.printLeaves(root.root.TAssoc)
        print("-------")
        root.printLeaves(root.root.TAssoc.TAssoc)

        range1 = [1, 2]
        print("Query:")
        L = self.one_d_range_query(root.root, range1)
        for i in L:
            print(i.coordinate)

        print("a")

    def build_range_tree(self, nodelist, flag, root):
        p_left_list = []
        p_right_list = []
        median = int(math.ceil(len(nodelist) / 2))
        # Construction of the associate Tree treeAssoc:
        t_assoc = None
        next_dim_node_list = []
        if len(nodelist) >= 2:
            if nodelist[0].nextDimNode is not None:
                for i in nodelist:
                    next_dim_node_list.append(i.nextDimNode)
                # Sorting the list
                next_dim_node_list.sort(key=operator.attrgetter('coordinate'))
                t_assoc = self.build_range_tree(next_dim_node_list, flag,root)

        if len(nodelist) == 1:
            # Base case of the recursion
            mid = Node(nodelist[0].dimension, nodelist[0].point, None, nodelist[0].dynamicPoint)
            mid.setNextDimNode(nodelist[0].nextDimNode)
        else:
            if flag == 0:
                median = int(math.ceil(len(self.nodeXList) / 2))
                root.setRoot(self.nodeXList[median-1])
                # The flag is used to recognise the very first recursion and set the root node
                flag +=1
            # Create two new subsets of points: <=median and >median
            count = 1
            for i in nodelist:
                if count <= median:
                    p_left_list.append(i)
                else:
                    p_right_list.append(i)
                count += 1

            mid = Node(nodelist[median - 1].dimension, nodelist[median - 1].point, nodelist[median - 1].TAssoc, nodelist[median - 1].dynamicPoint)
            mid.setChildren(nodelist[median - 1].rightChild, nodelist[median - 1].leftChild)
            mid.setNextDimNode(nodelist[median - 1].nextDimNode)
            if len(nodelist)> 1:
                # Make the TAssoc the associate structure of mid
                mid.TAssoc = t_assoc
            del t_assoc
            v_left = self.build_range_tree(p_left_list, flag, root)
            v_right = self.build_range_tree(p_right_list, flag, root)

            # The internal nodes are used only to guide the search path
            # For the leaves we create new Node classes
            if len(p_left_list) < 2:
                x = Node(v_left.dimension, v_left.point, v_left.TAssoc, v_left.dynamicPoint)
                x.setNextDimNode(v_left.nextDimNode)
                mid.leftChild = x
            else:
                mid.leftChild = v_left
            if len(p_right_list) < 2:
                y = Node(v_right.dimension, v_right.point, v_right.TAssoc, v_right.dynamicPoint)
                y.setNextDimNode(v_right.nextDimNode)
                mid.rightChild = y
            else:
                mid.rightChild = v_right
        return mid

    def find_split_node(self, root, xl, xr):
        # Input: A range tree T with two values xL and xR with xL <= xR
        # Output: The node v where the paths to xL and xR split, or the leaf where both paths end.
        v = root
        while (v.leftChild is not None) and (v.rightChild is not None) and\
                ((xr <= v.coordinate) | (xl > v.coordinate)):
            if xr <= v.coordinate:
                v = v.leftChild
            else:
                v = v.rightChild
        return v

    def dimensional_range_query(self, root, range):
        # Input: A dimensional range tree T and a range [x : x'] * [y : y'] * ...
        # Output: All points in T that lie in that range
        reported_list = []
        counter = 0
        while counter < len(range)-2:
            v_split = self.find_split_node(root, range[counter], range[counter+1])
            root = v_split.TAssoc
            counter +=2
        # counter -=2

        # if vSplit is a leaf
        if (v_split.leftChild is None) and (v_split.rightChild is None):
            # Checking if the point stored at vSplit must be reported
            if (v_split.coordinate >= range[counter-2]) & (v_split.coordinate <= range[counter-1]):
                reported_list.append(v_split)
        else:
            # Follow the path to x and call oneDrangeQuery on the subtrees right of the path
            v = v_split.leftChild
            # While v is not a leaf
            while (v.leftChild is not None) and (v.rightChild is not None):
                if range[counter-2] <= v.coordinate:
                    temp = self.one_d_range_query(v.rightChild.TAssoc, [range[counter], range[counter+1]])
                    if temp is not None:
                        for i in temp:
                            reported_list.append(i)
                    else:
                        if (range[counter] <= v.rightChild.nextDimNode.coordinate) & \
                                (range[counter+1] >= v.rightChild.nextDimNode.coordinate):
                            reported_list.append(v.rightChild.nextDimNode)
                    v = v.leftChild
                else:
                    v = v.rightChild
            # Check if the point stored at v must be reported
            if (v.coordinate >= range[counter-2]) & (v.coordinate <= range[counter-1])\
                         &(range[counter] <= v.nextDimNode.coordinate)\
                         &(range[counter+1] >= v.nextDimNode.coordinate):
                    reported_list.append(v.nextDimNode)

            # Similarly, follow the path from the right of vSplit to x',
            # call oneDRangeQuery with the range [y:y'] on the associate structures
            # of subtrees left of the path, and check if the point stored at the leaf
            # where the paths ends must be reported
            v = v_split.rightChild
            # While v is not a leaf

            while (v.leftChild is not None) and (v.rightChild is not None):
                if range[counter-1] >= v.coordinate:
                    temp = self.one_d_range_query(v.leftChild.TAssoc, [range[counter], range[counter+1]])
                    if temp is not None:
                        for i in temp:
                            reported_list.append(i)
                    else:
                        if (range[counter] <= v.leftChild.nextDimNode.coordinate) & \
                                   (range[counter+1] >= v.leftChild.nextDimNode.coordinate):
                            reported_list.append(v.leftChild.nextDimNode)
                    v = v.rightChild
                else:
                    v = v.leftChild

                # Check if the point stored at v must be reported
            if (v.coordinate >= range[counter-2]) & (v.coordinate <= range[counter-1])\
                        &(range[counter] <= v.nextDimNode.coordinate)\
                         &(range[counter+1] >= v.nextDimNode.coordinate):
                reported_list.append(v.nextDimNode)

        return reported_list

    def one_d_range_query(self, root, range):
        # Input: A binary search tree T and a range [x : x']
        # Output: All points stored in T that lie in that range

        if root is None:
            return

        reported_list = []
        v_split = self.find_split_node(root,range[0], range[1])
        if (v_split.leftChild is None) and (v_split.rightChild is None):
            # if vSplit is a leaf
            if (v_split.coordinate >= range[0]) & (v_split.coordinate <= range[1]):
                # Checking if the point stored at vSplit must be reported
                reported_list.append(v_split)
        else:
            # Follow the path to x(stored at range[0]) and report the points in subtrees right of the path
            v = v_split.leftChild
            # While v is not a leaf:
            while (v.leftChild is not None) and (v.rightChild is not None):
                if range[0] <= v.coordinate:
                    self.report_subtree(v.rightChild)
                    v = v.leftChild
                else:
                    v = v.rightChild
            # Check if the point stored at the leaf v must be reported:
            if (v.coordinate >= range[0]) & (v.coordinate <= range[1]):
                reported_list.append(v)

            # Similarly, follow the path to x'(stored at range[1]), report the points in subtrees left of the path,
            # and check if the point stored at the leaf where the path ends must be reported
            v = v_split.rightChild
            # While v is not a leaf:
            while (v.leftChild is not None) and (v.rightChild is not None):
                if range[1] >= v.coordinate:
                    self.report_subtree(v.leftChild)
                    v = v.rightChild
                else:
                    v = v.leftChild
            # Check if the point stored at the leaf v must be reported:
            if (v.coordinate >= range[0]) & (v.coordinate <= range[1]):
                reported_list.append(v)

        return reported_list

    def report_subtree(self, v):
        if v:
            if v.leftChild is None and v.rightChild is None:
                # First print the data of node only if it is a leaf
                print(v.coordinate)
            # Then recur on left child
            self.report_subtree(v.leftChild)
            # Finally recur on right child
            self.report_subtree(v.rightChild)


def main():
    test_range = RangeTree()
    test_range.initialization()


if __name__ == "__main__":
    main()