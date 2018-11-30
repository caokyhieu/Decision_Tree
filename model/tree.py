import numpy as np
import pandas as pd
from .node import *
import sys
sys.path.append('./utils')
from function import *


## Data input in the form of dataframe
class Tree:
	
	def __init__(self,data,classes,label,max_depth):
		self.head = Node(data.index,classes,label)
		self.leaves = [self.head]
		self.max_depth = max_depth
		
	
	def branch(self,leaf,data,label,attr): 
		leaf.feature = attr
		indexes = data.iloc[leaf.index].groupby([leaf.feature]).groups
		for name, index in indexes.items():
			classes = dict(data.iloc[index][label].value_counts())
			if len(classes) > 0:
				node = Node(index,classes,label,depth=leaf.depth+1)
				leaf.add_child(node,name)
				self.leaves.append(node)
		self.leaves.remove(leaf)
		
	
	def predict(self,data):
		predict = []
		for i in data.index:
			start = self.head
			while start.feature!=None:
				val = data.loc[i][start.feature]
				if val in start.valueNode.keys():
					start = start.valueNode[val]
				else:
					break
			predict.append(best_feature(start.classes))
		return np.array(predict)
	
	def calc_accuracy(self,predict,label):
		return (sum(label==predict)/len(label))