#!/usr/bin/python
#-*- coding:latin1 -*-
import pydot
import sys
import pandas as pd
import pyparsing
import numpy as np
import sklearn.cross_validation as cv
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn import metrics
from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.cross_validation import KFold



def ReadDataSet(path):

	""" Read the data set "zoo.csv", remove irrelevant variable, then split in (features,target) tuple for sklearn usage"""

	# read the dataset from the path file provided
	dataset = pd.read_csv(path,header=0)


	# Define the feature we will use, and remove irrelevant variables
	zoo_attribute=list(dataset.ix[:,(dataset.columns != 'type')].columns.values )
	zoo_attribute.remove('animal_name')
	features = dataset[zoo_attribute]

	# print "################################"
	# print "Features infromations\n"
	# print features.head()
	# print type(features)
	# print features.shape
	# print "################################\n"


	# Define the traget variable.
	target=dataset.type

	# print "################################"
	# print " Target infromations\n"
	# print target.head()
	# print type(target)
	# print target.shape
	# print "################################\n"

	return features,target

def PreprocessingData(X,Y,size):
	""" Peprocessing the data  before applying ML algorithm"""
	X_train, X_test, Y_train, Y_test = cv.train_test_split(X,Y,random_state=1,test_size=size)


	# print X_train.shape
	# print X_test.shape
	# print Y_train.shape
	# print Y_test.shape
	
	# Apply the min max normalization
	min_max_scaler = preprocessing.MinMaxScaler()
	X_train_normalized = min_max_scaler.fit_transform(X_train)
	X_test_normalized = min_max_scaler.fit_transform(X_test)

	return X_train, X_train_normalized, X_test, X_test_normalized, Y_train, Y_test

def Nnet(X_train,X_test,Y_train,Y_test,hiddenN,hiddenL,learning,momentum,iteration):

	clf = MLPClassifier(algorithm='sgd',alpha=10e-05,hidden_layer_sizes=(hiddenN,hiddenL),random_state=1,verbose=True, learning_rate_init=learning, momentum=momentum,max_iter=iteration)
	print clf
	clf.fit(X_train,Y_train)
	clf.predict(X_test)
	print "Error rate= "+str(1-metrics.accuracy_score(Y_test, clf.predict(X_test)))
	print "Mean squared error = "+str(metrics.mean_squared_error(Y_test, clf.predict(X_test)))
	print "Mean absolute error = "+str(metrics.mean_absolute_error(Y_test, clf.predict(X_test)))
	print "Median absolute error = "+str(metrics.median_absolute_error(Y_test, clf.predict(X_test)))
	return metrics.mean_squared_error(Y_test, clf.predict(X_test))


def DecisionTree(X_train,X_test,Y_train,Y_test):
	clf = tree.DecisionTreeClassifier(criterion="entropy")
	print clf
	clf.fit(X_train,Y_train)
	clf.predict(X_test)

	# Draw the decision tree a didacated pdf file
	YTarget=['Mamifere','Oiseau','Reptile','Poisson','Amphibien','Insecte','Invertebré']
	dot_data = StringIO() 
 	tree.export_graphviz(clf, out_file=dot_data,feature_names=list(X_train),class_names=YTarget) 
	graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
 	graph.write_pdf("tree.pdf") 

 	print "Error rate= "+str(1-metrics.accuracy_score(Y_test, clf.predict(X_test)))
	print "Mean squared error = "+str(metrics.mean_squared_error(Y_test, clf.predict(X_test)))
	print "Mean absolute error = "+str(metrics.mean_absolute_error(Y_test, clf.predict(X_test)))
	print "Median absolute error = "+str(metrics.median_absolute_error(Y_test, clf.predict(X_test)))



def main():

	print ""
		#############################################"
	print "IFT-7025-AI-Neural network script experimention"
	print "Houssem Sebouai"
	print "##############################################################################\n"
	
	print "Acquiring data set 'zoo.csv' "
	# Reading data-set
	dataset = ReadDataSet('zoo.csv')
	features = dataset[0]
	target = dataset[1]
	print "Data set loaded."
	print ""

	
	print "############################################"
	print "Begening the manual experimentation:"
	print ""
	""" #################### Neural Network Manual experimentation #################### """
	# Runing the manual experimentation

	# Request user parameter and puthem in array
	nb_exper=int(input("Number of experiment:"))
	print""
	param=[[] for i in range(nb_exper)]
	for i in range (len(param)):
		print "Pour le "+str(i+1)+"th experimetation donnez les parametre suivant"
		param[i].append(int(input("Hidden neurone:")))
		param[i].append(int(input("Hidden layer:")))
		param[i].append(float(input("Train proportion:")))
		param[i].append(float(input("Learning rate:")))
		param[i].append(float(input("Momentum:")))
		print "________________________________________"	
	# print param
	# print param.__len__()


	#preparing output file for the manual experimentation
	orig_stdout = sys.stdout
	f = file('manual.txt', 'w')
	sys.stdout = f

	Cumul_error=0
	for i in range (len(param)):
		print "Pour le "+str(i+1)+"th experimentaion, les resulats sont:"
		print "_________________________________________________________"
		print ""
		# Preprocessing data-set
		size=1-param[i][2]
		sets = PreprocessingData(features,target,size)

		# Using the neural network model
		hiddenN=param[i][0]
		hiddenL=param[i][1]
		learning=param[i][3]
		momentum=param[i][4]
		iteration=600
		Cumul_error += Nnet(sets[1],sets[3],sets[4],sets[5],hiddenN,hiddenL,learning,momentum,iteration)
		print Cumul_error
		print "___________Fin de l'experimentation de la "+str(i+1)+" experimentation___________________"
		print ""

	print 'Moyenne des erreurs quadratique des 10 est: '+ str(Cumul_error/10)
	# Closing ourput file rerouting stdout
	sys.stdout = orig_stdout
	f.close()	
	print "Manual experimenatation done."
	print ""

	print "############################################"
	print "Begening the K-folds experimentation:"
	print ""
	""" #################### Neural Network K-Fold experimentation #################### """
	# Runing the k-fold experimentation:
	kf=KFold(len(features),n_folds=10,shuffle=True)
	k_error=0

	#preparing output file for the k-folds experimentation
	orig_stdout = sys.stdout
	f = file('kfolds.txt', 'w')
	sys.stdout = f
	i=0
	for train, test in kf:
		i+=1
		print "Pour k=" +str(i)+ ", nous avons les resultats suivants"
		print "_________________________________________________________"
		print ""
		features_train,features_test= features.iloc[train], features.iloc[test]
		target_train,target_test= target.iloc[train], target.iloc[test]
		min_max_scaler = preprocessing.MinMaxScaler()
		features_train = min_max_scaler.fit_transform(features_train)
		features_test = min_max_scaler.fit_transform(features_test)
		k_error += Nnet(features_train,features_test,target_train,target_test,20,1,0.2,0.6,500)
		print "k_error=" +str(k_error)
		print "____________________Fin K="+str(i)+"________________________"
		print ""

	
	print 'Moyenne des k erreurs quadratique est: '+ str(k_error/10)

	# Closing ourput file rerouting stdout
	sys.stdout = orig_stdout
	f.close()
	print "K-Folds experimenatation done."
	print ""


	print "############################################"
	print "Begening the Decision Tree experimentation:"
	print ""
	""" #################### Decision Tree experimentation #################### """
	sets = PreprocessingData(features,target,0.25)
	# preparing output file for the Decision Tree experimentation
	orig_stdout = sys.stdout
	f = file('decisionTree.txt', 'w')
	sys.stdout = f

	DecisionTree(sets[0],sets[2],sets[4],sets[5])

	# Closing ourput file rerouting stdout
	sys.stdout = orig_stdout
	f.close()

	print "Decision Tree experimenatation done."
	print ""


if __name__ == '__main__':
	main()

