# coding=utf-8
from treelib import Tree, Node
from terminaltables import AsciiTable
import numpy as np


def direct_product(*args: iter):
	"""
	Calculates direct product of 2 or more iterables.
	:param args: must be iterables, total count varies
	:return: list of all possible combinations of the elements in given iterables
	"""
	res = list()
	for arg in args:
		if res:
			r = list()
			for i in arg:
				for rs in res:
					r.append([i, *rs])
			res = r
		else:
			for i in arg:
				res.append([i])
	return res


def pool(matrix: np.ndarray, a, b, c, d, total=None):
	"""
	When sudoku square contains elements in total, calculates all possible entries in the position [a,b,c,d]
	:param matrix: the Sudoku data matrix in 4 dims, (3,3,3,3)
	:param a: 1st dimension
	:param b: 2nd dimension
	:param c: 3rd dimension
	:param d: 4th dimension
	:param total: the complete set of possible elements, typically {1,2,3,4,5,6,7,8,9} in Sudoku
	:return: set of possible elements in position [a,b,c,d]
	"""
	if total is None:
		total = set(range(1, 10))
	value = matrix[a, b, c, d]
	matrix[a, b, c, d] = 0
	result = total \
			 - set(matrix[:, :, c, d].reshape(-1)) \
			 - set(matrix[a, b, :, :].reshape(-1)) \
			 - set(matrix[a, :, c, :].reshape(-1))
	matrix[a, b, c, d] = value
	return result


class Sudoku:
	"""
	Define sudoku puzzles
	"""

	def __init__(self, holes=0):
		self.data = np.zeros((3, 3, 3, 3), dtype='int')

		element = range(3)
		order = direct_product(element, element, element, element)

		i = 0
		genTree = Tree()
		root = Node(i, 'root', data=[order[0], self.data.copy()])
		genTree.add_node(root)
		currentNode = root
		getData = lambda node: node.data[1][tuple(node.data[0])]
		while i < len(order):
			i += 1
			a, b, c, d = order[i - 1]
			numPool = pool(self.data, a, b, c, d) - set(map(getData, genTree.children(currentNode.identifier)))
			if numPool:
				self.data[a, b, c, d] = np.random.choice(list(numPool))
				node = Node(i, data=[order[i - 1], self.data.copy()])
				genTree.add_node(node, currentNode)
				currentNode = node
			else:
				prev = genTree.parent(currentNode.identifier)
				while len(genTree.children(prev.identifier)) == len(pool(prev.data[1], *(prev.data[0]))):
					currentNode = prev
					prev = genTree.parent(currentNode.identifier)
				else:
					currentNode = prev
					self.data = currentNode.data[1].copy()
					i = currentNode.tag
				continue

		h = np.random.choice(len(order), size=holes, replace=False)
		self._answer = self.data.copy()
		self.holes = np.array(order)[h]
		self.data[tuple(self.holes.T.tolist())] = 0

	def __str__(self):
		data = np.array(self.data.reshape((9, 9)), dtype='str')
		data[data == '0'] = ' '
		table = AsciiTable(data.tolist())
		table.inner_row_border = True
		return table.table

	def answer(self):
		data = np.array(self._answer.reshape((9, 9)), dtype='str')
		data[data == '0'] = ' '
		table = AsciiTable(data.tolist())
		table.inner_row_border = True
		print(table.table)
		return self._answer


if __name__ == '__main__':
	s = Sudoku(holes=60)
	print(s)
	s.answer()
