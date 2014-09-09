#!/usr/bin/env python
#-*-coding:utf-8-*-

from numpy import *
import operator
import numpy as np
import os


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
def autoNorm(dataSet):
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals
	normDataSet = zeros(shape(dataSet))
	m = dataSet.shape[0]
	normDataSet = dataSet - tile(minVals,(m,1))
	normDataSet = normDataSet/tile(ranges,(m,1))
	return normDataSet,ranges,minVals



#定义测试函数，这里将数据集随机的划分为90% 和10%的两个子数据集，
#分别作为训练集和测试集
def datingClassTest():
	hoRatio = 0.10
	datingDataMat,datingLabels = file2matrix('../data/datingTestSet2.txt')
	normMat,ranges,minVals = autoNorm(datingDataMat)
	m = normMat.shape[0]
	numTestVecs = int(m*hoRatio)
	errorCount = 0.0
	for i in range(numTestVecs):
		classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],
				datingLabels[numTestVecs:m],3)
		print "the classifier come bacl with： %d,the real answer is %d" \
				%(classifierResult,datingLabels[i])
		if (classifierResult != datingLabels[i]):
			errorCount += 1
	print "the total error rate is: %f" % (errorCount/float(numTestVecs))





###########################################################################
#下面是手写数字的knn应用

#将图像转换为向量
def img2vector(filename):
	returnvect = zeros((1,1024))
	fr = open(filename)
	for i in range(32):
		linestr = fr.readline()
		for j in range(32):
			returnVect[0,32j+1] = int[lineStr[j]]
	return returnVect


#下面是手写数据的测试代码
def handwritingClassTest():
	hwLabels = []
	#获取目录内容
	trainFileList = os.listdir('../data/trainingDigits')
	m = len(trainingFileList)
	trainingMat = zeros((m,1024))
	for i in range(m):
		#从文件名解析分类数字
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		hwLabels.append(classNumStr)
		trainingMat[i,:] = img2vector('../data/trainingDigits/%s' % fileNameStr)

	testFileList = listdir('../data/testDigists')
	errorCount = 0.0
	mTest = len(testFileList)
	for i in range(mTest):
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		vectorUnderTest = img2rvctor('../data/testDigits/%s' % fileNameStr)
		classifierResult = classify0(vectorUnderTest,\
				trainingMat,heLabels,3)
		print 'the classifier came back with: %d ,the real answer is : %d' \
				%(classifierResult,classNumStr)
		if classifierResult != classNumStr: 
			errornum += 1
	print "\n the total number of errors is : %d" % errorCount
	print "\n the total rate is : %f" %(errorCount/float(mTest))
