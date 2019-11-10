import random

class Node:
    # the tree node in a binary(2-Ary) tree
    def __init__(self, key,value, left, right):
        self.key = key
        self.left = left
        self.right = right
        self.value = value


class dNode(Node):
    """
    a special type of binary tree node with its depth in the tree
    """

    def __init__(self, key,value, left, right, depth=0):
        Node.__init__(self, key, value,left, right)
        self.depth = depth


class WAVLTree():
    def __init__(self):
        # WAVLTree.__init__(self)
        self.root = None

    def insert(self, key,value):
        """
        calls recursive function _insert() for insertion.
        _insert() can be implemented by variations of the
        binary search tree.
        """
        self.root = self._insert(key,value, self.root)

    def _insert(self, key,value, node):
        if node == None:
            return dNode(key, value,None, None, 1)
        else:
            if node.key > key:
                node.left = self._insert(key,value, node.left)
                return self._insert_rotate(node, node.left, node.right)
            elif node.key < key:
                node.right = self._insert(key,value, node.right)
                return self._insert_rotate(node, node.right, node.left)
            else:
                return node


    def search(self, key, node):
        if node == None:
            return 
            
        else:
            if node.key > key:
                return self.search(key, node.left)
            elif node.key < key:
                return self.search(key, node.right)
            else:
                return node

    def find(self,key):
        x = self.search(key,self.root)
        return x

    def remove(self, key):
        """
        calls recursive function _remove() for remokey.
                _remove() can be implemented by variations of the
                binary search tree.
        """
        self.root = self._remove(key, self.root)

    def _remove(self, key, node):
        if node == None:
            return None
        else:
            if node.key == key:
                if node.left != None and node.right != None:
                    closest = self._find_leftmost(node.right)
                    self._swap_node_key(closest, node)
                    node.right = self._remove(key, node.right)
                    return self._remove_rotate(node, \
                                               node.right, node.left)
                elif node.left != None:
                    return node.left
                elif node.right != None:
                    return node.right
                else:
                    return None
            elif node.key < key:
                node.right = self._remove(key, node.right)
                return self._remove_rotate(node, node.right, \
                                           node.left)
            else:
                node.left = self._remove(key, node.left)
                return self._remove_rotate(node, node.left, \
                                           node.right)

    def _insert_rotate(self, parent, target, sibling):
        # calculate the rank difference of target node
        rd = self._rank_diff(parent, target)
        if rd == 0:
            # calculate the rank difference of sibling node
            rs = self._rank_diff(parent, sibling)
            if rs == 1:
                # if sibling has a rank difference of 1, we can safely
                # promote parent's rank and preserve the rank-diiferenc
                # property among parent, target and sibling by creating
                # a 1-2 node
                parent.depth += 1
                return parent
            elif rs == 2:
                # when sibling has a rank difference of 2,
                # promotion is not safe since it will increase
                # sibling's rank difference to 3. Therefore,
                # we need a trinode rotation
                return self._trinode_rotate(parent)
        elif rd == 1:
            # when target's rank difference is 1, then everything
            # is just perfect. just return the parent node.
            return parent

    def _remove_rotate(self, parent, target, sibling):
        rd = self._rank_diff(parent, target)
        if target == None and sibling == None:
            # parent has two external nodes
            parent.depth = 1
            return parent
        elif rd == 3:
            # when target has rank difference of 3
            rs = self._rank_diff(parent, sibling)
            if rs == 2:
                # when sibling has rank difference of 2
                # safe to demote parent's rank so that
                # rank_diff(parent,target)==2 and
                # rank_diff(parent,sibling)==1.
                parent.depth -= 1
                return parent
            else:
                # otherwise, sibling has rank difference of 1
                rsl = self._rank_diff(sibling, sibling.left)
                rsr = self._rank_diff(sibling, sibling.right)
                if rsl == 2 and rsr == 2:
                    # if sibling's children both have rank
                    # difference of 2, safe to demotion
                    # parent and sibling so that rank
                    # difference of target is 1 and
                    # rank difference of sibling is 1.
                    parent.depth -= 1
                    sibling.depth -= 1
                    return parent
                else:
                    return self._trinode_rotate(parent)
        else:
            return parent

    def _rank_diff(self, np, nq):
        return self._depth(np) - self._depth(nq)

    def _depth(self, node):
        if node == None:
            return 0
        else:
            return node.depth

    def _LL_rotate(self, node):
        n_parent = node.left
        node.left = n_parent.right
        n_parent.right = node
        self.set_depth(node)
        self.set_depth(n_parent)
        return n_parent

    def _RR_rotate(self, node):
        n_parent = node.right
        node.right = n_parent.left
        n_parent.left = node
        self.set_depth(node)
        self.set_depth(n_parent)
        return n_parent

    def _trinode_rotate(self, node):
        n_parent = node
        if self._depth(node.left) - self._depth(node.right) > 1:
            if self._depth(node.left.right) > \
                    self._depth(node.left.left):
                n_parent.left = self._RR_rotate(node.left)
            return self._LL_rotate(n_parent)
        elif self._depth(node.right) - self._depth(node.left) > 1:
            if self._depth(node.right.left) > \
                    self._depth(node.right.right):
                n_parent.right = self._LL_rotate(node.right)
            return self._RR_rotate(n_parent)
        else:
            self.set_depth(n_parent)
            return n_parent

    def set_depth(self, n):
        if n != None:
            n.depth = max(self._depth(n.left) + 1, self._depth(n.right) + 1)

    def _find_leftmost(self, node):
        """
        find the leftmost node in the subtree rooted at node.
        while-loop alternative (a):
            while (node.left!=None):
                node=node.left
            return node
        """
        if node.left == None:
            return node
        else:
            return self._find_leftmost(node.left)

    def _swap_node_key(self, src_node, dst_node):
        """
        swap the key of the source node and the one of the destination        node.
        """
        temp = src_node.key
        value = src_node.value
        src_node.key = dst_node.key
        src_node.value = dst_node.value
        dst_node.key = temp
        dst_node.value=value

    def inorder_traverse(self, node, action):
        if node != None:
            self.inorder_traverse(node.left, action)
            action(node)
            self.inorder_traverse(node.right, action)

    def preorder_traverse(self, node, action):
        if node != None:
            action(node)
            self.preorder_traverse(node.left, action)
            self.preorder_traverse(node.right, action)

    def postorder_traverse(self, node, action):
        if node != None:
            self.postorder_traverse(node.left, action)
            self.postorder_traverse(node.right, action)
            action(node)

    def printR(self, node):
        if node != None:
            self.printR(node.left)
            print("*" * node.depth, node.key,node.value)
            self.printR(node.right)

    def print(self, node, height=0):
        """
        a general function that prints out the tree rooted at node.
        """
        if node != None:
            self.print(node.left, height + 1)
            print("*" * height, node.key)
            self.print(node.right, height + 1)


if __name__ == "__main__":
    random.seed(10)
    w = WAVLTree()
    for x in range(20):
        w.insert(x,[34,23,113,133])
    print("Before remove key, Rank ")
    w.printR(w.root)
    print("\nTree")
    w.remove(12)
    print("Before remove key, Rank ")
    w.printR(w.root)
    print("\nTree")
    x =  w.find(5)
    print(x.key)
    print(x.value)
    x.value[2]=25
    print(x.value)
