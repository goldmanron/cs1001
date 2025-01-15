# Skeleton file for HW5 - Winter 2025 - extended intro to CS

# Add your implementation to this file

# You may add other utility functions to this file,
# but you may NOT change the signature of the existing ones.

# Change the name of the file to include your ID number (hw5_ID.py).
import math
import random


#####################################
# Linked List   (code from lecture) #
#####################################

class Node:
    def __init__(self, val):
        self.value = val
        self.next = None

    def __repr__(self):
        return str(self.value)


class Linked_list:
    def __init__(self, seq=None):
        self.head = None
        self.size = 0
        if seq != None:
            for x in seq[::-1]:
                self.add_at_start(x)

    def __repr__(self):
        out = ""
        p = self.head
        while p != None:
            out += p.__repr__() + ", "
            p = p.next
        return "[" + out[:-2] + "]"  # discard the extra ", " at the end

    def add_at_start(self, val):
        ''' add node with value val at the list head '''
        tmp = self.head
        self.head = Node(val)
        self.head.next = tmp
        self.size += 1

    def __len__(self):
        ''' called when using Python's len() '''
        return self.size

    def index(self, val):
        ''' find index of (first) node with value val in list
            return None of not found '''
        p = self.head
        i = 0  # we want to return the location
        while p != None:
            if p.value == val:
                return i
            else:
                p = p.next
                i += 1
        return None  # in case val not found

    def __getitem__(self, i):
        ''' called when reading L[i]
            return value of node at index 0<=i<len '''
        assert 0 <= i < len(self)
        p = self.head
        for j in range(0, i):
            p = p.next
        return p.value

    def __setitem__(self, i, val):
        ''' called when using L[i]=val (indexing for writing)
            assigns val to node at index 0<=i<len '''
        assert 0 <= i < len(self)
        p = self.head
        for j in range(0, i):
            p = p.next
        p.value = val
        return None

    def insert(self, i, val):
        ''' add new node with value val before index 0<=i<=len '''
        assert 0 <= i <= len(self)
        if i == 0:
            self.add_at_start(val)  # makes changes to self.head
        else:
            p = self.head
            for j in range(0, i - 1):  # get to position i-1
                p = p.next
            # now add new element
            tmp = p.next
            p.next = Node(val)
            p.next.next = tmp
            self.size += 1

    def append(self, val):
        self.insert(self.size, val)

    def pop(self, i):
        ''' delete element at index 0<=i<len '''
        assert 0 <= i < len(self)
        if i == 0:
            self.head = self.head.next  # bypass first element
        else:  # i >= 1
            p = self.head
            for j in range(0, i - 1):
                p = p.next

            # now p is the element BEFORE index i
            p.next = p.next.next  # bypass element at index i

        self.size -= 1


##############################################
# Binary Search Tree     (code from lecture) #
##############################################

class TreeNode():
    def __init__(self, key, val=None):
        self.key = key
        self.val = val
        self.left = None
        self.right = None

    def __repr__(self):
        return "(" + str(self.key) + ":" + str(self.val) + ")"


class BinarySearchTree():
    def __init__(self, root=None, size=0):
        self.root = root
        self.size = size

    def __repr__(self):  # you don't need to understand the implementation of this method
        def printree(root):
            if not root:
                return ["#"]

            root_key = str(root.key)
            left, right = printree(root.left), printree(root.right)

            lwid = len(left[-1])
            rwid = len(right[-1])
            rootwid = len(root_key)

            result = [(lwid + 1) * " " + root_key + (rwid + 1) * " "]

            ls = len(left[0].rstrip())
            rs = len(right[0]) - len(right[0].lstrip())
            result.append(ls * " " + (lwid - ls) * "_" + "/" + rootwid * " " + "\\" + rs * "_" + (rwid - rs) * " ")

            for i in range(max(len(left), len(right))):
                row = ""
                if i < len(left):
                    row += left[i]
                else:
                    row += lwid * " "

                row += (rootwid + 2) * " "

                if i < len(right):
                    row += right[i]
                else:
                    row += rwid * " "

                result.append(row)

            return result

        return '\n'.join(printree(self.root))

    def lookup(self, key):
        ''' return value of node with key if exists, else None '''
        node = self.root
        while node != None:
            if key == node.key:
                return node.val  # found!
            elif key < node.key:
                node = node.left
            else:
                node = node.right
        return None

    def insert(self, key, val):
        ''' insert node with key,val into tree.
            if key already there, just update its value '''

        parent = None  # this will be the parent of the new node
        node = self.root

        while node != None:  # keep descending the tree
            if key == node.key:
                node.val = val  # update the val for this key
                return

            parent = node
            if key < node.key:
                node = node.left
            else:
                node = node.right

        if parent == None:  # was empty tree, need to update root
            self.root = TreeNode(key, val)
        elif key < parent.key:
            parent.left = TreeNode(key, val)  # "hang" new node as left child
        else:
            parent.right = TreeNode(key, val)  # "hang"    ...     right child

        self.size += 1
        return None

    def minimum(self):
        ''' return value of node with minimal key '''

        if self.root == None:
            return None  # empty tree has no minimum
        node = self.root
        while node.left != None:
            node = node.left
        return node.val

    def depth(self):
        ''' return depth of tree, uses recursion '''

        def depth_rec(node):
            if node == None:
                return -1
            else:
                return 1 + max(depth_rec(node.left), depth_rec(node.right))

        return depth_rec(self.root)


##############
# QUESTION 1 #
##############
class LLLNode:
    def __init__(self, val):
        self.next_list = []
        self.val = val

    def __repr__(self):
        st = "Value: " + str(self.val) + "\n"
        st += "Neighbors:" + "\n"
        for p in self.next_list:
            st += " - Node with value: " + str(p.val) + "\n"
        return st[:-1]


class LogarithmicLinkedList:
    def __init__(self):
        self.head = None
        self.len = 0

    def __len__(self):
        return self.len

    def add_at_start(self, val):
        node = LLLNode(val)
        if len(self) == 0:
            self.head = node
            self.len = 1
            return None

        current = self.head
        level = 0
        while current and level < int(math.log2(len(self))) + 1:
            node.next_list.append(current)
            if level < len(current.next_list):
                current = current.next_list[level]
            else:
                current = None
            level += 1

        self.head = node
        self.len += 1
        return None

    def __getitem__(self, i):
        current = self.head
        j = 0  # Start from the first level (0th level)
        while j < len(current.next_list) and (1 << j) <= i:
            j += 1
        j -= 1  # Go back to the last valid level

        while j >= 0:
            if (1 << j) <= i:
                current = current.next_list[j]
                i -= (1 << j)  # Reduce `i` by the jump size
            j -= 1

        return current.val

    # Optional - improve this code!
    def __contains__(self, val):
        p = self.head
        while p:
            if p.val == val:
                return True

            # Find the largest pointer in next_list with a value <= val
            level = len(p.next_list) - 1
            while level >= 0 and p.next_list[level].val > val:
                level -= 1

            # If no valid pointer exists, terminate the search
            if level < 0:
                break

            # Move to the next node in the list
            p = p.next_list[level]

        return False


##############
# QUESTION 2 #
##############

def gen1():
    n = 0
    while True:
        for x in range(-n, n + 1):
            for y in range(-n, n + 1):
                if abs(x) + abs(y) == n:  # Ensures points are generated layer by layer
                    yield x, y
        n += 1


def gen2(g):
    s = 0
    while True:
        s += next(g)
        yield s


def gen3(g):
    pass  # replace this with your code (or don't, if there does not exist such generator with finite delay)


def gen4(g):
    inc = True
    dic = True
    prev = next(g)
    while True:
        yield inc or dic
        curr = next(g)
        inc, dic = inc and prev <= curr, dic and prev >= curr
        prev = curr


def gen5(g1, g2):
    pass  # replace this with your code (or don't, if there does not exist such generator with finite delay)


def gen6(g1, g2):
    while True:
        a, b = next(g1), next(g2)
        if a != b:
            yield a
            yield b


def gen7():
    i = 1
    while True:
        yield gen7_helper(i)
        i += 1


def gen7_helper(i):
    j = 0
    while True:
        yield j * i
        j += 1


##############
# QUESTION 3 #
##############

def lowest_common_ancestor(t, n1, n2):
    def lowest_common_ancestor_helper(tree):
        if tree.key in {n1, n2}:
            return tree

        if tree.left.key and n1 <= tree.left.key <= n2 < tree.key:
            return lowest_common_ancestor_helper(tree.left)

        if tree.right.key and tree.key < n1 <= tree.right.key <= n2:
            return lowest_common_ancestor_helper(tree.right)

        return tree

    return lowest_common_ancestor_helper(t.root)


def build_balanced(n):
    def build_balanced_rec(d, k):
        t = TreeNode(2 ** (d - 1) + k)
        if d > 1:
            t.left, t.right = build_balanced_rec(d - 1, k), build_balanced_rec(d - 1, k + 2 ** (d - 1))

        return t

    return BinarySearchTree(build_balanced_rec(n, 0), 2 ** n - 1)


def subtree_sum(t, k):
    return (2 ** (int(math.log(k)) + 1) - 1) * k


##############
# QUESTION 4#
##############
def prefix_suffix_overlap(lst, k):
    return [(i, j) for i in range(len(lst)) for j in range(len(lst)) if i != j and lst[i][:k] == lst[j][-k:]]


class Dict:
    def __init__(self, m, hash_func=hash):
        """ initial hash table, m empty entries """
        self.table = [[] for i in range(m)]
        self.hash_mod = lambda x: hash_func(x) % m

    def __repr__(self):
        L = [self.table[i] for i in range(len(self.table))]
        return "".join([str(i) + " " + str(L[i]) + "\n" for i in range(len(self.table))])

    def insert(self, key, value):
        """ insert key,value into table
            Allow repetitions of keys """
        i = self.hash_mod(key)  # hash on key only
        item = [key, value]  # pack into one item
        self.table[i].append(item)

    def find(self, key):
        """ returns ALL values of key as a list, empty list if none """
        i = self.hash_mod(key)
        return [item[1] for item in self.table[i] if item[0] == key]


def prefix_suffix_overlap_hash1(lst, k):
    n = len(lst)
    d = Dict(n)
    for i in range(n):
        d.insert(lst[i][:k], i)

    return [(i, j) for j in range(n) for i in d.find(lst[j][-k:]) if i != j]


def almost_prefix_suffix_overlap_hash1(lst, k):
    n = len(lst)

    d = Dict(n)

    for i in range(n):
        pref = lst[i][:k]
        for j in range(k):
            pref1 = ''.join([pref[m] if m != j else '*' for m in range(k)])
            d.insert(pref1, i)

    l = []
    memo = set((i, i) for i in range(n))
    for p in range(k):
        for i in range(n):
            suff = lst[i][-k:]
            for j in d.find(''.join([suff[m] if m != p else '*' for m in range(k)])):
                if (i, j) not in memo:
                    l += [(j, i)]
                    memo.add((i, j))
    return list(l)


##############
# QUESTION 5 #
##############


class Rational1:
    """ represent a rational number using nominator and denominator """

    def __init__(self, n, d):
        assert isinstance(n, int) and isinstance(d, int)
        g = math.gcd(n, d)
        self.n = n // g  # nominator
        self.d = d // g  # denominator

    def __repr__(self):
        if self.d == 1:
            return "<Rational " + str(self.n) + ">"
        else:
            return "<Rational " + str(self.n) + "/" + str(self.d) + ">"

    def is_int(self):
        return self.d == 1

    def floor(self):
        return self.n // self.d

    def __eq__(self, other):
        if isinstance(other, Rational1):
            return self.n == other.n and self.d == other.d
        elif isinstance(other, int):
            return self.n == other and self.d == 1
        else:
            return False

    def __add__(self, other):
        return Rational1(self.n * other.d + other.n * self.d, self.d * other.d)

    def __mul__(self, other):
        return Rational1(self.n * other.n, self.d * other.d)

    def inverse(self):
        return Rational1(self.d, self.n)

    def divides(self, other):
        return (self * other.inverse()).is_int() if other.n != 0 else False

    def __lt__(self, other):
        if self.d * other.d > 0:
            return self.n * other.d < other.n * self.d

        return self.n * other.d > other.n * self.d


class Rational2:
    """ represent a rational number using quotient, remainder and denominator """

    def __init__(self, n, d):
        assert isinstance(n, int) and isinstance(d, int)
        g = math.gcd(n, d)
        n, d = n // g, d // g
        self.q = n // d  # quotient
        self.r = n % d  # remainder
        self.d = d  # denominator

    def __repr__(self):
        if self.r == 0:
            return "<Rational " + str(self.q) + ">"
        else:
            n = self.q * self.d + self.r
            return "<Rational " + str(n) + "/" + str(self.d) + ">"

    def is_int(self):
        return self.r == 0

    def floor(self):
        return self.q

    def __eq__(self, other):
        if isinstance(other, Rational2):
            return self.q == other.q and \
                self.r == other.r and \
                self.d == other.d
        elif isinstance(other, int):
            return self.q == other and self.r == 0
        else:
            return False

    def __add__(self, other):
        return Rational2((self.q * self.d + self.r) * other.d + (other.q * other.d + other.r) * self.d,
                         self.d * other.d)

    def __mul__(self, other):
        return Rational2((self.q * self.d + self.r) * (other.q * other.d + other.r), self.d * other.d)

    def inverse(self):
        return Rational2(self.d, self.q * self.d + self.r)

    def divides(self, other):
        return (self * other.inverse()).is_int() if (self.q * self.d + self.r) != 0 else False

    def __lt__(self, other):
        if self.d * other.d > 0:
            return (self.q * self.d + self.r) * other.d < (other.q * other.d + other.r) * self.d

        return (self.q * self.d + self.r) * other.d > (other.q * other.d + other.r) * self.d


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):
        return self.x < other.x or (self.x == other.x and self.y < other.y)
