from treelib import Tree,Node
from terminaltables import AsciiTable
import numpy as np
import torch
import torch.nn as nn


def direct_product(*args:iter):
	res=list()
	for arg in args:
		if res:
			r=list()
			for i in arg:
				for rs in res:
					r.append([i,*rs])
			res=r
		else:
			for i in arg:
				res.append([i])
	return res

def pool(matrix:np.ndarray,a,b,c,d):
	return set(range(1,10))-set(matrix[:,:,c,d].reshape(-1))-set(matrix[a,b,:,:].reshape(-1))-set(matrix[a,:,c,:].reshape(-1))

class Sudoku:
	def __init__(self,holes=0):
		self.data=np.zeros((3,3,3,3),dtype='int')
		
		I=set(range(1,10))
		element=range(3)
		order=direct_product(element,element,element,element)
		
		i=0
		genTree=Tree()
		root=Node(i,'root',data=[order[0],self.data.copy()])
		genTree.add_node(root)
		currentNode=root
		getData=lambda node:node.data[1][tuple(node.data[0])]
		while i<len(order):
			i+=1
			a,b,c,d=order[i-1]
			numPool=pool(self.data,a,b,c,d)-set(map(getData,genTree.children(currentNode.identifier)))
			if numPool:
				self.data[a,b,c,d]=np.random.choice(list(numPool))
				node=Node(i,data=[order[i-1],self.data.copy()])
				genTree.add_node(node,currentNode)
				currentNode=node
			else:
				prev=genTree.parent(currentNode.identifier)
				while len(genTree.children(prev.identifier))==len(pool(prev.data[1],*(prev.data[0]))):
					currentNode=prev
					prev=genTree.parent(currentNode.identifier)
				else:
					currentNode=prev
					self.data=currentNode.data[1].copy()
					i=currentNode.tag
				continue
		
		h=np.random.choice(len(order),size=holes,replace=False)
		self._answer=self.data.copy()
		self.holes=np.array(order)[h]
		self.data[tuple(self.holes.T.tolist())]=0
	
	def __str__(self):
		data=np.array(self.data.reshape((9,9)),dtype='str')
		data[data=='0']=' '
		table=AsciiTable(data.tolist())
		table.inner_row_border=True
		return table.table
	
	def answer(self):
		data=np.array(self._answer.reshape((9,9)),dtype='str')
		data[data=='0']=' '
		table=AsciiTable(data.tolist())
		table.inner_row_border=True
		print(table.table)
		return self._answer


class Solver(torch.nn.Module):
	def __init__(self):
		super(Solver, self).__init__()
		self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
		self.bn3 = nn.BatchNorm2d(32)

		# Number of Linear input connections depends on output of conv2d layers
		# and therefore the input image size, so compute it.
		def conv2d_size_out(size, kernel_size = 5, stride = 2):
			return (size - (kernel_size - 1) - 1) // stride  + 1
		convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
		convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
		linear_input_size = convw * convh * 32
		self.head = nn.Linear(linear_input_size, outputs)

	# Called with either one element to determine next action, or a batch
	# during optimization. Returns tensor([[left0exp,right0exp]...]).
	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		return self.head(x.view(x.size(0), -1))

if __name__=='__main__':
	s=Sudoku(holes=60)
	print(s)
	s.answer()