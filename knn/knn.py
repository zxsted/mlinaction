#!/usr/bin/env python
#-*-coding:utf-8-*-

from numpy import *
import operator
import numpy as np

import matplotlib
import matplotlib.pyplot as plt


#定义数据集
def createDataSet():
	group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels = ['A','A','B','B']
	return group ,labels

#knn算法
def classify0(inX,dataSet,labels,k):
	'''
	@param inX:用于分类的输入向量
	@param dataet: 输入的训练样本
	@param labels：标签向量
	@param k：最近邻的数目
	'''
	dataSetSize = dataSet.shape[0]
	#计算距离
	diffMat = tile(inX,(dataSetSize,1)) - dataSet  #tile ：Construct an array by repeating A the number of times given by reps.
	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances**0.5
	sortedDistIndicies = distances.argsort()

	#选择距离最小的k个点
	classCount = {}
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
	#对结果进行排序返回前k个
	sortedClassCount = sorted(classCount.iteritems(),key = operator.itemgetter(1),reverse=True)
	return sortedClassCount[0][0]


#使用k-nn算法改进约会网站的配对效果

#将文本数据转换为numpy数组
def file2matrix(filename):
	fr = open(filename)
	arrayOLines = fr.readlines()
	numberOfLines = len(arrayOLines)
	returnMat = zeros((numberOfLines,3))
	classLabelVector = []
	index = 0
	for line in arrayOLines:
		line = line.strip()   #截取调所有的回车字符
		listFromLine = line.split('\t')
		returnMat[index,:] = listFromLine[0:3]
		classLabelVector.append(int(listFromLine[-1]))
		index += 1
	return returnMat,classLabelVector

#绘制数据的散点图，观察数据的分布样式

def data2graph(dataMat,labels,index):
	fig = plt.figure()
	axes = fig.add_subplot(111)
	axes.scatter(dataMat[:,index[0]],dataMat[:,index[1]],
			15.0*np.array(labels),15.0*np.array(labels))
	plt.show()



#归一化特征值
def autoNorm(dataset):
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals
	normDataSet = zeros(shape(dataSet))
	normDataSet = dataSet - tile(minVals,(m,1))
	normDataSet = normDataSet/tile(ranges,(m,1))
	return normDataSet,ranges,minVals




