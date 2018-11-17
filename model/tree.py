import numpy as np
import pandas as pd
import node

class Tree:
	
	def __init__(self,data,classes,label):
		self.head = Node(data.index,classes,label)
		self.leaves = [self.head]
		self.time_branch = 0
		
	
	def branch(self,leaf,data,label,attr): 
		leaf.feature = attr
		leaf.values = set(data.iloc[leaf.index][leaf.feature])
		indexes = data.iloc[leaf.index].groupby([leaf.feature]).groups
		for name, index in indexes.items():
			classes = list(data.iloc[index][label].unique())
			self.leaves.append(leaf.subnode(name,index,classes,label))
		self.leaves.remove(leaf)
		self.time_branch +=1