from tree import *

def build_tree(data,features,label):
	classes = list(data[label].unique())
	tree = Tree(data,classes,label)
	pos=0
	
	while pos<len(tree.leaves):
		leaf = tree.leaves[pos]
		
		if (len(leaf.classes)<=1) or (len(data.iloc[leaf.index])<=1):
			leaf.feature = leaf.classes[0]
			pos+=1
		
		else:
			dic_gain = {}
			for feature in features:
				dic_gain[feature] = calc_info_gain(data[[feature,label]],label,feature)
			name = ''
			max_ = -1
			
			for i ,j in dic_gain.items():
				if j>max_:
					max_ = j
					name = i
			best_feature = name
			tree.branch(leaf,data,label,best_feature)
			
	return tree