import numpy as np
import pandas as pd


def calc_entropy(data):
	values = set(data)
	count = {}
	for value in values:
		count[value] = 0
	for i in data:
		count[i]+=1
	counts = np.array(list(count.values()))
	p =list(counts/sum(counts))
	e = 0
	for i in p:
		if i!=0:
			e += -i * np.log2(i)     
	return e

def calc_info_gain(data,label,feature):
	entropyS = data[[label]].apply(calc_entropy)
	featureCount = data.groupby([feature]).count().values[:,0]
	weights = featureCount/sum(featureCount)
	en = sum(data.groupby([feature])[label].apply(calc_entropy)* weights)
	gain = entropyS - en
	return gain[0]