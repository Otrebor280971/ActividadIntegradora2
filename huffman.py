"""
===============================================================
 Program : 05_huffman.py
 Author  : Juan Manuel Ahuactzin
 Date    : 2025-10-13
 Version : 1.0
===============================================================
Description:
    Implementation of Huffman coding using a min-heap-based 
    approach to build the optimal binary prefix tree.
    Each node represents a character and its frequency, and 
    the resulting tree provides the Huffman code for each symbol.
    The class includes a __str__ method to visualize the tree
    structure and binary code paths (0/1).

	Adapted from: GeeksforGeeks. (s. f.). Huffman Coding | Greedy 
	Algo-3. GeeksforGeeks. 
	https://www.geeksforgeeks.org/dsa/huffman-coding-greedy-algo-3/

Usage:
    python 05_huffman.py

Example:
    python 05_huffman.py

Dependencies:
    - Python 3.x
    - Standard libraries: heapq

Notes:
    - The program constructs the Huffman tree and prints its
      hierarchical representation, followed by the Huffman code
      assigned to each character.
    - The __str__ method allows visualization of the recursive 
      structure, indicating '0' for left edges and '1' for right 
      edges.
===============================================================
"""


import heapq

# Class to represent huffman tree


class Node:
    def __init__(self, x, character=""):
        self.data = x
        self.char = character
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.data < other.data

    def __str__(self, level=0, prefix="Root: ", cumchain=""):
        """Devuelve una representación jerárquica del árbol."""
        result = "   " * level + \
            f"{prefix}({self.char}:{self.data}):{cumchain}\n"
        if self.left:
            result += self.left.__str__(level + 1,
                                        prefix="L-0- ", cumchain=cumchain+'0')
        if self.right:
            result += self.right.__str__(level + 1,
                                         prefix="R-1- ", cumchain=cumchain+'1')
        return result

# Function to traverse tree in preorder
# manner and push the huffman representation
# of each character.


def preOrder(root, ans, curr):
    if root is None:
        return

    # Leaf node represents a character.
    if root.left is None and root.right is None:
        ans[root.char] = curr
        return

    preOrder(root.left, ans, curr + '0')
    preOrder(root.right, ans, curr + '1')


def huffmanCodes(s, freq):
    # Code here
    n = len(s)

    # Min heap for node class.
    pq = []
    for i in range(n):
        tmp = Node(freq[i], s[i])
        heapq.heappush(pq, tmp)

    # Construct huffman tree.
    while len(pq) >= 2:
        # Left node
        l = heapq.heappop(pq)

        # Right node
        r = heapq.heappop(pq)

        newNode = Node(l.data + r.data)
        newNode.left = l
        newNode.right = r

        heapq.heappush(pq, newNode)

    root = heapq.heappop(pq)
    codes_dic = {}
    preOrder(root, codes_dic, "")
    return codes_dic, root


if __name__ == "__main__":
    s = ["a", "b", "c", "d", "e", "f"]
    freq = [5, 9, 12, 13, 16, 45]
    codes_dic, root = huffmanCodes(s, freq)
    print(root)

    codes_dic = dict(sorted(codes_dic.items()))

    for c in codes_dic:
        print(f"{c}={codes_dic[c]}")
