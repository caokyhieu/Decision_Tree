from .tree import *
from sklearn.metrics import f1_score
from .node import *
import pandas as pd
import numpy as np
import sys

sys.path.append('./utils')

from function import *



def build_tree(data,features,label,max_depth,prune_threshold =5e-4,func=calc_entropy):
	classes = dict(data[label].value_counts())
	tree = Tree(data,classes,label,max_depth)
	position = 0
	while position <len(tree.leaves):

		leaf = tree.leaves[position]
		if (len(leaf.classes)<=1) or (len(data.iloc[leaf.index])<=1) or (leaf.depth >=tree.max_depth):
			position += 1
		else:
			dic_gain = {}
			for feature in features:
				info_gain = calc_info_gain(data[[feature,label]],label,feature,func)
				if (info_gain >prune_threshold) and (data.iloc[leaf.index][[feature]].apply(func)[0]!=0):
					dic_gain[feature] = info_gain
					
			if len(dic_gain) >0:
				bFeature = best_feature(dic_gain)
				tree.branch(leaf,data,label,bFeature)
			else:
				position += 1
	return tree



def cross_validation(df,label,k_fold,max_depth,threshold,func):
	df = df.sample(frac=1).reset_index(drop=True)
	size = int(len(df)/k_fold)
	result_val = []
	for i in tqdm(range(1,k_fold)):
		
		if i < k_fold-1:
			index_train = [x for x in range(len(df)) if x< i*size or x >=(i+1)*size]
			index_test = [x for x in range(i*size,(i+1)*size)]
		else:
			index_train = [x for x in range(len(df)) if x<i*size ]
			index_test = [x for x in range(i*size,len(df))]
		
		new_test_df = df.iloc[index_test].reset_index(drop=True)
		new_train_df = df.iloc[index_train].reset_index(drop=True)    
		tree = build_tree(new_train_df,set(new_train_df.columns)-{label},label,max_depth,threshold,func)
		pos_label = list(set(tree.head.classes)- {best_feature(tree.head.classes)})
		result_val.append(f1_score(tree.predict(new_test_df),new_test_df[label],pos_label= pos_label[0]))

	return np.mean(result_val)


def grid_search(train_df,label,k_fold=5,max_depth=range(3,7),prune_threshold=np.arange(0,0.1,0.01),func=[calc_entropy,calc_gini]):
	param_grid = {'max_depth': max_depth, 'prune_threshold' :prune_threshold ,'func':func}

	grid = ParameterGrid(param_grid)

	max_acc = 0.0
	selected_depth = 0
	selected_prune_threshold = 0.0
	selected_func = calc_entropy
	# i = 0
	for params in grid:
	# i+=1
	# print(str(i/80*100)+'%')
		acc = cross_validation(train_df,label,k_fold,params['max_depth'],params['prune_threshold'],params['func'])
		if acc > max_acc:
			max_acc = acc
			selected_depth = params['max_depth']
			selected_prune_threshold = params['prune_threshold']
			selected_func = params['func']
		
	return(max_acc, selected_depth, selected_prune_threshold,selected_func)