#!/usr/bin/env python
#-*-encoding:utf-8-*-

from math import log

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
			reducedFeadVec.extend(featVec[axis+1:])
			retDataSet.append(reducedFeatVec)
	return retDataSet


#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
	numFeatures = len(dataSet[0]) - 1
