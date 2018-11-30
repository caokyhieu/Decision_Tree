import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

from graphviz import Digraph,Source




def best_feature(dic):
	max_ = np.max(list(dic.values()))
	for key,value in dic.items():
		if value == max_:
			return key
	return None

def calc_entropy(data):
	count = np.array(data.value_counts())
	p = count/np.sum(count)
	e = np.dot(-p , np.log2(p))   
	return e

def calc_gini(data):
	count = np.array(data.value_counts())
	p = count/np.sum(count)
	e = 1 - np.dot(p,p)   
	return e

def calc_info_gain(data,label,feature,func=calc_entropy):
	S = data[[label]].apply(func)
	featureCount = data.groupby([feature]).count().values[:,0]
	weights = featureCount/sum(featureCount)
	sub_S = np.dot(data.groupby([feature])[label].apply(func), weights)
	gain = S - sub_S
	return gain[0]

def reset_index(node):
	if len(node.index)==0:
		return
	else:
		node.index = []
		node.error_rate = 1
		for i in node.valueNode.values():
			reset_index(i)
			

def calc_error_rate(node,data,label):
	node.index = data.index
	bestClass = best_feature(node.classes)
	if(bestClass in data[label].value_counts().index):
		node.error_rate = 1 - data[label].value_counts()[bestClass]/len(node.index)
	else:
		node.error_rate = 1
	if len(node.valueNode)==0:
		return
	else:
		for name,index in data.groupby([node.feature]).groups.items():
			if name in node.valueNode.keys():
				calc_error_rate(node.valueNode[name],data.loc[index],label)
			else:
				continue


# Prune tree by Reduced Error Pruning

def remove_leaf(tree,node):
	
	child_error_rate = 0
	for child in node.valueNode.values():
		child_error_rate +=  child.error_rate*sum(child.classes.values())
	child_error_rate /= sum(node.classes.values())
	if(node.error_rate<=child_error_rate):
		for i in node.valueNode.values():
			tree.leaves.remove(i)
		tree.leaves.append(node)
		node.valueNode = {}
		node.values = None
		node.feature = None
	else:
		return
	
def prune_tree(tree,node):
	if (node.valueNode):
		decide = True
		for child_ in node.valueNode.values():
			if child_ not in tree.leaves :
				decide =False
		if (decide):
			remove_leaf(tree,node)
		   
			
		else:
			for child in node.valueNode.values():
				prune_tree(tree,child)
	return tree


## Draw the tree
def draw_graph(root):
	dot = Digraph(comment='Decision Tree')
	parent = []
	dot.node('%s'%(root),'%s\n %s'%(root.feature,root.classes))
	parent.append(root)
	
	dot.attr('node', shape='oval')
	while len(parent)>0:
		
		element = parent[0]
		parent = parent[1:]
		if len(element.valueNode)>0:
			for i,sub in element.valueNode.items():
				if (sub.feature!=None):
					dot.node('%s'%(sub),'%s\n %s'%(sub.feature,sub.classes))
				else:
					
					dot.node('%s'%(sub),'%s'%(str(best_feature(sub.classes))))
				dot.edge('%s'%(element),'%s'%(sub),label=str(i))
				parent.append(sub)
		else:
			continue
	return dot
