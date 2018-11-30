import numpy as np
import pandas as pd

class Node:
	def __init__(self,index,classes,label,depth=0,attribute=None):
		self.feature = attribute
		self.index = index
		self.classes = classes
		self.valueNode = {}
		self.depth = depth
		
		self.error_rate = 1
		
	def add_child(self, node,value):
		self.valueNode[value] = node