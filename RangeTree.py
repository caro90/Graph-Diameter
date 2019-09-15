from DataHandler import *
from binaryTrees import *
import operator
import math


class RangeTree:
    # TODO: write doc for each function and class
    """
    The main class that allows the creation of a Range Tree with multiple dimension support.
    It builds the Range Tree using a recursive divide and conquer approach. It also allows range queries
    where reports all the points under a given range.

    """

    # TODO: Check if this init is relevant and delete
    def __init__(self):

        self.dimension = 1
        self.nodeXList = []
        self.dynamicList = []


    def initialization(self):
        """
        Loading points and creating Node objects that will be used to build the Range Tree.
        Typically, for a given point (x,y,z) three Node objects are being created, one for each
        dimension.

        Returns:
        """

        root = BST(None)
        x = PointHandler()
        list_of_points = x.insertFile_XYZval("points4.txt")
        # length_of_indices: saves the number of indices/coordinates
        length_of_indices = len(list_of_points[0].pointList)

        # Creating Node objects for all the coordinates (x, y, z ...)
        for i in list_of_points:
            temp = Node(1, i, None)
            self.nodeXList.append(temp)
            for j in range(2, length_of_indices):
                temp.setNextDimNode(Node(j, i, None))
                temp = temp.nextDimNode

        # Sorting according to x coordinate
        self.nodeXList.sort(key=operator.attrgetter('coordinate'))

        # TESTING:
        x = self.build_range_tree(self.nodeXList, 0, root)
        root.setRoot(x)

        print("---Post Order----")
        root.printPostorder(root.root.TAssoc.TAssoc.TAssoc.TAssoc)
        print("----1st Dim---")
        root.printLeaves(root.root)
        print("---2nd Dim----")
        root.printLeaves(root.root.TAssoc)
        print("----3rd Dim---")
        root.printLeaves(root.root.TAssoc.TAssoc)
        print("----4rd Dim---")
        root.printLeaves(root.root.TAssoc.TAssoc.TAssoc)
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
        range3 = [-3, -5, -11, 8, -100, 120]
        L3 = self.dimensional_range_query(root.root, range3)
        for j in L3:
            print(j.coordinate)

        print("-----------")
        print("Query 5D:")
        range3 = [-5, 200, -100, 100, -100, 120, -10, 100, -5, 55]
        L3 = self.dimensional_range_query(root.root, range3)
        for j in L3:
            print(j.coordinate)

    #TODO: integrate initialization2 to initialization and delete
    def initialization2(self, pi):
        """
        TODO: better documentation
        Loading points and creating Node objects that will be used to build the Range Tree.
        Typically, for a given point (x,y,z) three Node objects are being created, one for each
        dimension.

        Returns:
        """

        root = BST(None)
        list_of_points = []
        x = PointHandler()
        for i in pi:
            point = x.insertManually2([i])
            list_of_points.append(point)

        # Creating Node objects for all the coordinates
        # For a point: (x, y, z) we create 3 Node objects.
        for i in list_of_points:
            temp = Node(1, i, None)
            self.nodeXList.append(temp)
            print(len(i.pointList))
            for j in range(2, len(i.pointList)+1):
                temp.setNextDimNode(Node(j, i, None))
                temp = temp.nextDimNode


        # Sorting according to the first coordinate
        self.nodeXList.sort(key=operator.attrgetter('coordinate'))
        # TESTING:
        x = self.build_range_tree(self.nodeXList, 0, root)
        root.setRoot(x)

        """
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
        """

        return root

    def build_range_tree(self, nodelist, flag, root):
        """
        Using a divide and conquer approach builds the Range Tree.

        Parameters
        ----------
            nodelist:

            flag:

            root

        Returns:
             The median node of the first coordinate
        """
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
                t_assoc = self.build_range_tree(next_dim_node_list, flag, root)

        if len(nodelist) == 1:
            # Base case of the recursion
            mid = Node(nodelist[0].dimension, nodelist[0].point, None)
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

            mid = Node(nodelist[median - 1].dimension, nodelist[median - 1].point, nodelist[median - 1].TAssoc)
            mid.setChildren(nodelist[median - 1].rightChild, nodelist[median - 1].leftChild)
            mid.setNextDimNode(nodelist[median - 1].nextDimNode)
            if len(nodelist) > 1:
                # Make the TAssoc the associate structure of mid
                mid.TAssoc = t_assoc
            del t_assoc
            v_left = self.build_range_tree(p_left_list, flag, root)
            v_right = self.build_range_tree(p_right_list, flag, root)

            # The internal nodes are used only to guide the search path
            # For the leaves we create new Node classes
            if len(p_left_list) < 2:
                x = Node(v_left.dimension, v_left.point, v_left.TAssoc)
                x.setNextDimNode(v_left.nextDimNode)
                mid.leftChild = x
            else:
                mid.leftChild = v_left
            if len(p_right_list) < 2:
                y = Node(v_right.dimension, v_right.point, v_right.TAssoc)
                y.setNextDimNode(v_right.nextDimNode)
                mid.rightChild = y
            else:
                mid.rightChild = v_right
        return mid

    def find_split_node(self, root, xl, xr):
        """
         Parameters
         ----------
            A range tree T with two values xL and xR with xL <= xR

         Returns
         -------
             The node v where the paths to xL and xR split, or the leaf where both paths end.
        """

        v = root
        if v is None:
            return
        while (v.leftChild is not None) and (v.rightChild is not None) and\
                ((xr <= v.coordinate) | (xl > v.coordinate)):
            if xr <= v.coordinate:
                v = v.leftChild
            else:
                v = v.rightChild


        return v

    def dimensional_range_query(self, root, range):
        """
         Input: A dimensional range tree T and a range [x : x'] * [y : y'] * ...
         Returns:
              All points in T that lie in that range
        """

        reported_list = []
        counter = 0
        while counter < len(range)-2:
            v_split = self.find_split_node(root, range[counter], range[counter+1])
            root = v_split.TAssoc
            counter +=2

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
        """
         Input: A binary search tree T and a range [x : x']
         Returns:
            All points stored in T that lie in that range
        """

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
        """
        """
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