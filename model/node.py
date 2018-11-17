import numpy as np
import pandas as pd

class Node:
	def __init__(self,index,classes,label,attribute=None,valueNames=None):
		self.feature = attribute
		self.values = valueNames
		self.index = index
		self.classes = classes
		self.valueNode = {}
		
	def subnode(self,value,index,classes,label):
		self.valueNode[value] = Node(index,classes,label)
		return  self.valueNode[value]