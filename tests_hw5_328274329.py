import unittest
from hw5_328274329 import LogarithmicLinkedList, lowest_common_ancestor, build_balanced, subtree_sum, prefix_suffix_overlap, BinarySearchTree, TreeNode

class TestLogarithmicLinkedList(unittest.TestCase):
    def setUp(self):
        self.lll = LogarithmicLinkedList()

    def test_add_at_start(self):
        self.lll.add_at_start(1)
        self.assertEqual(self.lll.head.val, 1)
        self.lll.add_at_start(2)
        self.assertEqual(self.lll.head.val, 2)
        self.assertEqual(self.lll.head.next_list[0].val, 1)

    def test_len(self):
        self.assertEqual(len(self.lll), 0)
        self.lll.add_at_start(1)
        self.assertEqual(len(self.lll), 1)
        self.lll.add_at_start(2)
        self.assertEqual(len(self.lll), 2)

    def test_getitem(self):
        self.lll.add_at_start(1)
        self.lll.add_at_start(2)
        self.lll.add_at_start(3)
        self.assertEqual(self.lll[0], 3)
        self.assertEqual(self.lll[1], 2)
        self.assertEqual(self.lll[2], 1)

    def test_contains(self):
        self.lll.add_at_start(1)
        self.lll.add_at_start(2)
        self.lll.add_at_start(3)
        self.assertTrue(1 in self.lll)
        self.assertTrue(2 in self.lll)
        self.assertTrue(3 in self.lll)
        self.assertFalse(4 in self.lll)

class TestLowestCommonAncestor(unittest.TestCase):
    def setUp(self):
        self.tree = BinarySearchTree(TreeNode(20))
        self.tree.insert(10, None)
        self.tree.insert(30, None)
        self.tree.insert(5, None)
        self.tree.insert(15, None)
        self.tree.insert(25, None)
        self.tree.insert(35, None)

    def test_lca(self):
        self.assertEqual(lowest_common_ancestor(self.tree, 5, 15).key, 10)
        self.assertEqual(lowest_common_ancestor(self.tree, 5, 30).key, 20)
        self.assertEqual(lowest_common_ancestor(self.tree, 25, 35).key, 30)

class TestBuildBalanced(unittest.TestCase):
    def test_build_balanced(self):
        tree = build_balanced(3)
        self.assertEqual(tree.root.key, 4)
        self.assertEqual(tree.root.left.key, 2)
        self.assertEqual(tree.root.right.key, 6)
        self.assertEqual(tree.root.left.left.key, 1)
        self.assertEqual(tree.root.left.right.key, 3)
        self.assertEqual(tree.root.right.left.key, 5)
        self.assertEqual(tree.root.right.right.key, 7)

class TestSubtreeSum(unittest.TestCase):
    def setUp(self):
        self.tree = build_balanced(3)

    def test_subtree_sum(self):
        self.assertEqual(subtree_sum(self.tree, 1), 7)
        self.assertEqual(subtree_sum(self.tree, 2), 14)
        self.assertEqual(subtree_sum(self.tree, 3), 21)

class TestPrefixSuffixOverlap(unittest.TestCase):
    def test_prefix_suffix_overlap(self):
        lst = ["abc", "bca", "cab"]
        self.assertEqual(prefix_suffix_overlap(lst, 1), [(0, 1), (1, 2), (2, 0)])
        self.assertEqual(prefix_suffix_overlap(lst, 2), [])

if __name__ == '__main__':
    unittest.main()
