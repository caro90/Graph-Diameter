
class Node:
    """
    
    """
    # TODO: make _init_ simpler with only one point, change dimension to be passed and not assign here
    def __init__(self, dimension, point, TAssoc, dynamicPoint):
        self.nextDimNode = None
        self.point = point
        self.dynamicPoint = dynamicPoint
        self.dimension = dimension
        self.TAssoc = TAssoc
        # val is not strictly defined yet
        if point is not None:
            if self.dimension == 1: self.coordinate = int(point.x)
            if self.dimension == 2: self.coordinate = int(point.y)
            if self.dimension == 3:
                # if dimension is the last one we also store the value of the point
                self.coordinate = int(point.z)
                self.val = int(point.val)
        else:
            # Sets the coordinate of the current node from the dimension
            self.coordinate = self.dynamicPoint.listOfCoordinates[dimension-1]
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


''' def insertNodes(self, data):
        if self.data:
            if data < self.data:
                if self.left is None:
                    self.left = Node(data)
                else:
                    self.left.insert(data)
            elif data > self.data:
                if self.right is None:
                    self.right = Node(data)
                else:
                    self.right.insert(data)
        else:
            self.data = data
'''


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