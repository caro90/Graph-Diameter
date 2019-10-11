from DataHandler import *
from binaryTrees import *
import operator
import math


class RangeTree:
    """
    The main class that allows the creation of a Range Tree with multiple dimension support.


    This class supports a function for building the Range Tree using a recursive divide and conquer approach.
    It also allows range queries where reports all the points under a given range.

    """

    def __init__(self):
        self.nodeXList = []

    def initialization(self):
        """
        Loads points and creates Node objects that will be used to build the Range Tree.
        Typically, for a given point (x,y,z) three Node objects are being created, one for each
        dimension. This function is mainly used for the testing of the range tree, so the points are
        being imported from a file. To create a range tree that will later be used by another function use
        def initialization2

        """
        root = BST(None)
        x = PointHandler()
        list_of_points = x.read_file("points3D.txt")
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
        x = self.build_range_tree(self.nodeXList, 0)
        root.setRoot(x)
        print("Query 3D:")
        range3 = [-5, 11, -11, 200, -100, 200]
        L3 = self.dimensional_range_query(root.root, range3, 0)
        for j in L3:
            print(j)
        print()

    def initialization2(self, pi):
        """
        Loading points and creating Node objects that will be used to build the Range Tree.
        Typically, for a given point (x,y,z) three Node objects are being created, one for each
        dimension.

        Parameters
        ----------
        pi: list of integers
            The numerical values of the points in each dimension

        Returns
        --------
        root: BST object
            A binary search tree which stores as an instance variable a Node object which is the root of the range tree
            at dimension 1. It also supports pre-order, post-order and  in-order printing.

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
            for j in range(2, len(i.pointList)):
                temp.setNextDimNode(Node(j, i, None))
                temp = temp.nextDimNode

        # Sorting according to the first coordinate
        self.nodeXList.sort(key=operator.attrgetter('coordinate'))
        # TESTING:
        x = self.build_range_tree(self.nodeXList, 0)
        root.setRoot(x)

        return root

    def build_range_tree(self, nodelist, flag):
        """
        This function builds a range tree or as often called a multi-level data structure. The main tree T is called
        first-level tree, and the associate structures are called second-level trees and so on if the dimension is
        bigger than 2. A recursive procedure is followed for the construction of the trees (in essence is a divide
        and conquer approach)

        Parameters
        ----------
        nodelist: list of Node objects.
              Each Node object has an instance variable called coordinate. In this variable the x-coordinate of the
              imported points is stored. The Node objects are stored in an increasing order on the x-coordinate.
        flag: int
            A flag to recognise the very first recursion.

        Returns
        -------
        mid: Node object
             The root node of the first coordinate
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
                t_assoc = self.build_range_tree(next_dim_node_list, flag)

        if len(nodelist) == 1:
            # Base case of the recursion
            mid = Node(nodelist[0].dimension, nodelist[0].point, None)
            mid.setNextDimNode(nodelist[0].nextDimNode)
        else:
            if flag == 0:
                median = int(math.ceil(len(self.nodeXList) / 2))
                flag += 1
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
            v_left = self.build_range_tree(p_left_list, flag)
            v_right = self.build_range_tree(p_right_list, flag)

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
        This function is used from the range query functions as a subroutine in order to find the node at which
        the paths of a given range of values is split

         Parameters
         ----------
         root: Node object
            The root of the tree.
         xl:
            A value which is <= xr
         xr:
            A value which is >= xl
         Returns
         -------
         v: Node object
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

    def dimensional_range_query(self, root, range, counter):
        """
        The main recursive function to perform a range query for a range tree of dimension >=2. It uses as subroutines
        the function find_split_node and the one_d_range_query.

         Parameters
         ----------
         root: Node object
             The root of the tree.
         range:
             The range of the values for each dimension in which the query will take place.
         counter: int
             It is used in each recursion to figure out the dimension and therefore the right range
             of values to perform the query.

         Returns
         -------
         reported_list: list of integers
              All points in T that lie in that range
        """
        # Base case of the recursion:
        reported_list = []
        if root is None:
            return

        v_split = self.find_split_node(root, range[counter], range[counter + 1])

        if (v_split.leftChild is None) and (v_split.rightChild is None):
            # if vSplit is a leaf
            # Checking if the point stored at vSplit must be reported
            if (v_split.coordinate >= range[counter]) & (v_split.coordinate <= range[counter+1]):
                if counter + 2 >= len(range):
                    return self.one_d_range_query(v_split, [range[counter], range[counter + 1]])
                else:
                    temp = self.dimensional_range_query(v_split.nextDimNode, range, counter + 2)

                if not temp:
                    return
                else:
                    return temp
        else:
            # Follow the path to x and call oneDrangeQuery on the subtrees right of the path
            v = v_split.leftChild
            # While v is not a leaf
            while (v.leftChild is not None) and (v.rightChild is not None):
                if range[counter] <= v.coordinate:
                    if counter + 2 < len(range):
                        if v.rightChild.TAssoc is not None:
                            temp = self.dimensional_range_query(v.rightChild.TAssoc, range, counter +2)
                            # if temp is not None:
                            #     reported_list.append(temp)
                        else:
                            temp = self.dimensional_range_query(v.rightChild.nextDimNode, range, counter + 2)
                            # if temp is not None:
                            #     reported_list.append(temp)
                    else:
                        temp = self.one_d_range_query(v.rightChild, [range[counter], range[counter+1]])
                        for i in temp:
                            reported_list.append(temp.pop())

                    if temp is not None and temp:
                        for i in temp:
                            reported_list.append(temp.pop())
                    v = v.leftChild
                else:
                    v = v.rightChild
            if counter + 2 < len(range):
                # Check if the point stored at v must be reported
                if (v.coordinate >= range[counter]) & (v.coordinate <= range[counter+1]):
                    temp = self.dimensional_range_query(v.nextDimNode, range, counter+2)
                    if temp is not None and temp:
                        for i in temp:
                            reported_list.append(temp.pop())

            # Similarly, follow the path from the right of vSplit to x',
            # call oneDRangeQuery with the range [y:y'] on the associate structures
            # of subtrees left of the path, and check if the point stored at the leaf
            # where the paths ends must be reported
            v = v_split.rightChild
            # While v is not a leaf
            while (v.leftChild is not None) and (v.rightChild is not None):
                if range[counter+1] >= v.coordinate:

                    if counter +2 < len(range):
                        if v.leftChild.TAssoc is not None:
                            temp = self.dimensional_range_query(v.leftChild.TAssoc, range, counter+2)
                        else:
                            temp = self.dimensional_range_query(v.leftChild.nextDimNode, range, counter + 2)
                    else:
                        temp = self.one_d_range_query(v.leftChild, [range[counter], range[counter+1]])

                    if temp is not None and temp:
                        for i in temp:
                            reported_list.append(temp.pop())
                    v = v.rightChild
                else:
                    v = v.leftChild

            if counter + 2 < len(range):
                # Check if the point stored at v must be reported
                if (v.coordinate >= range[counter]) & (v.coordinate <= range[counter+1]):
                    temp = self.dimensional_range_query(v.nextDimNode, range, counter+2)
                    if temp is not None and temp:
                        for i in temp:
                            reported_list.append(temp.pop())
        return reported_list

    def one_d_range_query(self, root, range):
        """
        It works in exactly the same fashion as dimensional_range_query but in one dimension.

        Parameters
        ----------
        root: Node object
             The root of the tree.
        range:
             The range of the values in one dimension in which the query will take place.

        Returns
        -------
        reported_list: list of integers
            All points in T that lie in that range
        """
        if root is None:
            return

        reported_list = []
        v_split = self.find_split_node(root,range[0], range[1])
        if (v_split.leftChild is None) and (v_split.rightChild is None):
            # if vSplit is a leaf
            if (v_split.coordinate >= range[0]) & (v_split.coordinate <= range[1]):
                # Checking if the point stored at vSplit must be reported
                #reported_list.append(v_split.val)
                self.report_subtree(v_split, reported_list)
        else:
            # Follow the path to x(stored at range[0]) and report the points in subtrees right of the path
            v = v_split.leftChild
            # While v is not a leaf:
            while (v.leftChild is not None) and (v.rightChild is not None):
                if range[0] <= v.coordinate:
                    # self.report_subtree(v.rightChild)
                    # reported_list.append(v.rightChild.val)
                    self.report_subtree(v.rightChild, reported_list)
                    v = v.leftChild
                else:
                    v = v.rightChild
            # Check if the point stored at the leaf v must be reported:
            if (v.coordinate >= range[0]) & (v.coordinate <= range[1]):
                #reported_list.append(v.val)
                self.report_subtree(v, reported_list)

            # Similarly, follow the path to x'(stored at range[1]), report the points in subtrees left of the path,
            # and check if the point stored at the leaf where the path ends must be reported
            v = v_split.rightChild
            # While v is not a leaf:
            while (v.leftChild is not None) and (v.rightChild is not None):
                if range[1] >= v.coordinate:
                    self.report_subtree(v.leftChild, reported_list)
                    #reported_list.append(v.leftChild.val)
                    v = v.rightChild
                else:
                    v = v.leftChild
            # Check if the point stored at the leaf v must be reported:
            if (v.coordinate >= range[0]) & (v.coordinate <= range[1]):
                #reported_list.append(v.val)
                self.report_subtree(v, reported_list)

        if len(reported_list) == 0:
            return
        else:
            return reported_list

    def report_subtree(self, v, reported_list):
        """
        This is a recursive function and is mainly used as subroutine from one_d_range query to report
        all the points under the given node.
        Parameters
        ---------
        v: Node object
            The node from which the function will start reporting its children including itself.
        reported_list: list of integers
            In this list are stored all the values.
        Returns
        -------
        Nothing to be returned as everything is stored in reported_list. Since reported_list is an object it is
        performed a call by reference.
        """
        if v:
            if v.leftChild is None and v.rightChild is None:
                # First print the data of node only if it is a leaf
                reported_list.append(v.val)
            # Then recur on left child
            self.report_subtree(v.leftChild, reported_list)
            # Finally recur on right child
            self.report_subtree(v.rightChild, reported_list)


def main():
    test_range = RangeTree()
    test_range.initialization()


if __name__ == "__main__":
    main()