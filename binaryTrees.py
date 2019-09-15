
class Node:
    """

    """

    def __init__(self, dimension, point, TAssoc):
        """

        """
        self.nextDimNode = None
        self.point = point
        self.dimension = dimension
        self.TAssoc = TAssoc
        if point is not None:
            self.coordinate = point.pointList[self.dimension-1]
            if self.dimension == len(point.pointList):
                # self.val is stored at the last position of every point: (x, y, z, val)
                self.val = point.pointList[len(point.pointList)-1]
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
    """

    """
    def __init__(self, root):
        # self.root: its class is Node
        self.root = root
        # associateT is a BST object reference to the associate tree of the current BST
        self.associateT = None

    def setRoot(self, root):
        self.root = root

    def setAssociateT(self, associateT):
        self.associateT = associateT

    def getRoot(self):
        return self.root

    def insertData(self, data):
        self.root.insertNodes(data)

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