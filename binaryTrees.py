
class Node:
    """
    The main object that is used from BST class and RangeTree class to create trees. For each one of the nodes
    of the tree on Node object is created.

    Parameters
    ----------
    point: Point object
        Every Node object stores a Point object in which are stored all the rest coordinates of the given point. For
        example if the Node object is storing x as its coordinate, then in the Point object it will be stored the whole
        initial point,namely (x, y, z), for the case of three dimensional point.
    dimension: int
        An integer to indicate to the Node object which coordinate to store as its own coordinate from the Point object.
    TAssoc: Node object
        Every Node object has a TAssoc(Associate tree) which points to the root node of the associate tree. In TAssoc
        are stored all the subsets of points in the leaves of the subtree rooted at the current Node object, but on the
        next dimension of the points. The associate tree structure is one of the main characteristics of RangeTree
        construction.

    Attributes
    -------------------
    nextDimNode: Node object
        It is just a reference to the node of the next dimension. Therefore for a point (x,y,z), the Node object that
        stores x stores at nextDimNode a reference to y Node, and y to z.
    coordinate: int
        Stores the value of x or y or z for the case of three dimensional point.
    leftChild: Node object
    rightChild: Node object
    """
    def __init__(self, dimension, point, TAssoc):
        self.nextDimNode = None
        self.point = point
        self.dimension = dimension
        self.TAssoc = TAssoc
        if point is not None:
            self.coordinate = point.pointList[self.dimension-1]
            if self.dimension == len(point.pointList)-1:
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
    A binary search tree object which provides some useful printing functions to manipulate the Node objects which
    are the nodes of the trees.

    Parameters
    ----------
    root: Node object
        It stores the root node of the tree.
    Attributes
    ----------
    associateT: BST
        A reference to the associate BST object in the next dimension. It is not used in the RangeTree class.
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