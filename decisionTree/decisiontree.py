#!/usr/bin/env python
#-*-encoding:utf-8-*-

from math import log
import operator

#产生一些测试数据
def createDataSet():
	dataSet = [[1,1,'yes'],
			[1,1,'yes'],
			[1,0,'no'],
			[0,1,'no'],
			[0,1,'no']]

	labels = ['not surfacing','flippers']
	return dataSet,labels

#计算给定数据集的香浓熵
def calcShannonEnt(dataSet):
	numEntries = len(dataSet)
	labelCounts = {}
	for featVec in dataSet:
		currentLabel  = featVec[-1]
		if currentLabel not in labelCounts:
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1


	shannonEnt =0.0
	for key in labelCounts:
		prob = float(labelCounts[key]) / numEntries
		shannonEnt -= prob * log(prob,2)
	return shannonEnt


#更具给定特征划分数据及
def splitDataSet(dataSet,axis,value):
	'''
	@param dataSet: 源数据集
	@param axis： 特征下标
	@param value： axis对应的特征包含的特征值
	@return retDataet ： axis特征的值为value的样本组成的特征向量
	'''
	retDataSet = []
	for featVec in dataSet:
		if featVec[axis] == value:   #将特征axis从特征向量中排除
			reducedFeatVec = featVec[:axis]
			reducedFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reducedFeatVec)
	return retDataSet


#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
	'''
	本函数要求数据满足以下两点要求：
	1、数据是由列表元素组成的列表，而且所有列元素都要具有相同的数据长度；
	2、数据的最后一列或者每个实例的最后一个元素是当前实例的类别标签。
	'''
	numFeatures = len(dataSet[0]) - 1
	baseEntropy = calcShannonEnt(dataSet)  #计算没有按特征划分前的熵
	bestInfoGain = 0.0
	bestFeature = -1
	for i in range(numFeatures):
		featList = [example[i] for example in dataSet]
		uniqueVals = set(featList)              #获取特征i的唯一分类标签列表
		newEntropy = 0.0
		#计算每种分类方式的信息熵
		for value in uniqueVals:
			subDataSet = splitDataSet(dataSet,i,value)
			prob = len(subDataSet) / float(len(dataSet))  #当前分类的的概率
			newEntropy += prob * calcShannonEnt(subDataSet) #累加各个分类的熵与本分类的乘积
		infoGain = baseEntropy - newEntropy
		#寻找最好的信息熵
		if (infoGain > bestInfoGain):
			bestInfoGain = infoGain
			gestFeatur = i
	return bestFeature


#选出投票最多的类作为返回值
def majorityCnt(classList):
	classCount={}
	for vote in classList:
		if vote not in classCount.keys():
			classCount[vote] = 0
			classCount[vote] += 1
		sortedClassCount = sorted(classCount.iteritems(),
				key=operator.item.getter(1),reverse=True)
	return sortedClassCount[0][0]


#创建树的代码
def createTree(dataSet,labels):
	classList = [example[-1] for example in dataSet]
	#如果当前类的所有样本都是一类则停止继续划分，并返回类别
	if classList.count(classList[0]) == len(classList):
		return classList[0]
	if len(dataSet[0]) == 1:      #即数据集的行长度为1，表明所有的特征都用完了，这时选择拥有实例最多的类
		return majorityCnt(classList)   #遍历完所有特征时返回出现次数最多的
	bestFeat = chooseBestFeatureToSplit(dataSet)
	print '最佳划分点是 %d'% bestFeat
	bestFeatLabel = labels[bestFeat]
	print '最佳划分特征是 %s'% bestFeatLabel
	myTree = {bestFeatLabel:{}}
	del(labels[bestFeat])
	print '去除最佳划分特征后的标签为'
	print '、'.join([label for label in labels])

	featValues = [example[bestFeat] for example in dataSet]  #获取所有实例对应的当前最优特征的数值 
	uniqueVals = set(featValues)                             #特征去重
	for value in uniqueVals:
		subLabels = labels[:]  #复制类标签 （因为函数参数是列表类型是，参数是按照引用方式传递的）
		myTree[bestFeatLabel][value] = createTree(splitDataSet\
				(dataSet,bestFeat,value),subLabels)
	return myTree



if __name__ == '__main__':
	myDat,labels = createDataSet()
	mytree = createTree(myDat,labels)
	print mytree
