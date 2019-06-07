class Node:
    def __init__(self,x,y,val):
        self.x = x
        self.y = y
        # val is not strictly defined yet
        self.val = val
        self.leftChild = None
        self.rightChild = None
        #if the node is not a leaf set up the next dimension
        if self.getChildren()!=None:
            self.nextDimension = BST()

    def get(self):
        return self.val

    def set(self, val):
        self.val = val

    def getChildren(self):
        children = []
        if (self.leftChild != None):
            children.append(self.leftChild)
        if (self.rightChild != None):
            children.append(self.rightChild)
        return children

    # A function to do inorder tree traversal
    def printInorder(self,root):
        if root:
            # First recur on left child
            self.printInorder(root.leftChild)

            # then print the data of node
            print(root.val),

            # now recur on right child
            self.printInorder(root.rightChild)

            # A function to do postorder tree traversal

    def printPostorder(self,root):

        if root:
            # First recur on left child
            self.printPostorder(root.leftChild)

            # the recur on right child
            self.printPostorder(root.rightChild)

            # now print the data of node
            print(root.val),

            # A function to do preorder tree traversal

    def printPreorder(self,root):

        if root:
            # First print the data of node
            print(root.val),

            # Then recur on left child
            self.printPreorder(root.leftChild)

            # Finally recur on right child
            self.printPreorder(root.rightChild)


class BST:
    def __init__(self):
        self.root = None

    def setRoot(self, x, y, val):
        self.root = Node(val)

    def insert(self, x, y, val):
        if (self.root is None):
            self.setRoot(val)
        else:
            self.insertNode(self.root, x, y, val)

    def insertNode(self, currentNode, x, y, val):
        if (val <= currentNode.val):
            if (currentNode.leftChild):
                self.insertNode(currentNode.leftChild, val)
            else:
                currentNode.leftChild = Node(val)
        elif (val > currentNode.val):
            if (currentNode.rightChild):
                self.insertNode(currentNode.rightChild, val)
            else:
                currentNode.rightChild = Node(val)

    def find(self, val):
        return self.findNode(self.root, val)

    def findNode(self, currentNode, val):
        if (currentNode is None):
            return False
        elif (val == currentNode.val):
            return True
        elif (val < currentNode.val):
            return self.findNode(currentNode.leftChild, val)
        else:
            return self.findNode(currentNode.rightChild, val)

    def printNode(self):
        return  self.root.printPostorder(self.root)


x = BST()
#x.insert(5)
#x.insert(8)
[x.insert(i) for i in range(1,10)]

print(x.printNode())