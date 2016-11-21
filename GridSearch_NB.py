
# This document is used to study various factors such as no of train documents, alpha, features vs accuracy on Train and cross Validation 
#sets, This uses a 10 fold stratified cv and uses grid search to implement the pipeline. Also plots various graphs.

import numpy as np
from BuildData import BuildData
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from nltk.tokenize import TreebankWordTokenizer
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import ShuffleSplit
# from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import learning_curve
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot as pl
from matplotlib.backends.backend_pdf import PdfPages
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_blobs
import gc
from sklearn import svm

file_loc='/Users/Dhanush/Desktop/Projects/DM_project/DMProj_Data/Data_train/CODE_'
file_loc_out ='/Users/Dhanush/Desktop/Projects/DM_project/DMProj_Data/Data_domain_selected/CODE_'

def data_extract_using_parms(Domain_select,write_output,characters_limit,output_loc,features):
	build_data_source =  BuildData(file_loc)
	build_data_source.extract_data_routines()
	data = build_data_source.fetch_train_test_data(Domain_select,write_output,characters_limit,output_loc)
	return data



def Vectorize_split(total_feature_list,total_label,features,binary):

	stop_words = {'english',}
	if features == 'max':
		cv = TfidfVectorizer(input ='total_feature_list',stop_words = {'english'},lowercase=True,analyzer ='word',binary =binary)#,non_negative=True)#,max_features =75000)
	else:
		cv = TfidfVectorizer(input ='total_feature_list',stop_words = {'english'},lowercase=True,analyzer ='word',binary =binary,max_features =features)#,norm='l2',sublinear_tf =True,min_df = 0.005)
	X = cv.fit_transform(total_feature_list)
	vocab = np.array(cv.get_feature_names())
	print len(vocab)
	#print vocab
	#feature_names = cv.get_feature_names()
	y = (np.array(total_label))
	train_test_data = [i for i in range(5)]
	#X_train, X_test, y_train, y_test = train_test_split(X,y ,test_size=0.2, random_state=5677)
	train_test_data[0] = X
	train_test_data[2] = y
	#train_test_data[0],train_test_data[1],train_test_data[2],train_test_data[3] = train_test_split(X,y ,test_size=0.2, random_state=5677)
	train_test_data[4] = len(vocab)
	return train_test_data
# Main routine, Create an instance of BuildData and call extract_data_routine

def MyMultiNomialNB(X_train, y_train):
	clf = MultinomialNB()
	#param_grid = {'alpha': [0.000000001,0.00000001,0.0000001,0.0000001] }
	param_grid = {'alpha': [0.1] }
	# Ten fold Cross Validation
	classifier= GridSearchCV(estimator=clf, cv=10 ,param_grid=param_grid)
	classifier.fit(X_train, y_train)
	return classifier.cv_results_

def MyBernoulliNB(X_train, y_train):
	clf = BernoulliNB()
	#param_grid = {'alpha': [0.000000001,0.00000001,0.0000001,0.0000001]}
	param_grid = {'alpha': [0.00001] }
	# Ten fold Cross Validation
	classifier= GridSearchCV(estimator=clf, cv=10 ,param_grid=param_grid)
	classifier.fit(X_train, y_train)
	return classifier.cv_results_

def MyGaussianNB(X_train, y_train):
	clf = GaussianNB()
	param_grid = {}
	# Ten fold Cross Validation
	classifier= GridSearchCV(estimator=clf, cv=3 ,param_grid=param_grid)
	classifier.fit(X_train, y_train)
	return classifier.cv_results_



# main Function
gc.enable()
#Varying Character limit and Document count and study Variance and Biasis
Domain_select = 0
write_output = 0
characters_limit = 10
Document_count = []
character_count =[]
results_CV_MNB = []
results_train_MNB = []
results_CV_GNB = []
results_train_GNB = []
results_CV_BNB = []
results_train_BNB = []
feature_count = []
features = 'max'
data=data_extract_using_parms(Domain_select,write_output,characters_limit,file_loc_out,features)
total_feature_list = data[0]
print len(total_feature_list)
total_label = data[1]
train_test_data=Vectorize_split(total_feature_list,total_label,features,False)
max_features =train_test_data[4]
print max_features
features = 1000
for i in range(1,2,1):
	#if features > max_features:
	#	continue
	"""
	print "Training loop: " + str(i)
	train_test_data=Vectorize_split(total_feature_list,total_label,features,True)
	results_BNB = MyBernoulliNB(train_test_data[0],train_test_data[2])
	#print MyBernoulliNB(train_test_data[0],train_test_data[2])
	results_train_BNB.append(1 - np.mean(results_BNB['mean_train_score']))
	results_CV_BNB.append(1-np.mean(results_BNB['mean_test_score']))
	"""
	train_test_data=Vectorize_split(total_feature_list,total_label,features,False)
	#results_MNB = MyMultiNomialNB(train_test_data[0],train_test_data[2])
	results_MNB = MyMultiNomialNB(train_test_data[0],train_test_data[2])
	results_train_MNB.append(np.mean(results_MNB['mean_train_score']))
	results_CV_MNB.append(np.mean(results_MNB['mean_test_score']))
	
	
	#results_GNB = MyGaussianNB(train_test_data[0].toarray(),train_test_data[2])
	#results_train_GNB.append(1 - np.mean(results_GNB['mean_train_score']))
	#results_CV_GNB.append(1-np.mean(results_GNB['mean_test_score']))
	
	#Document_count.append((train_test_data[0]).shape[0])
	#character_count.append(characters_limit) 
	feature_count.append(train_test_data[4])

print len(feature_count)
print len(results_MNB['mean_train_score'])
print "------- Multinomial Naive Bayes-------------------------------------"
index_max_accuracy = results_CV_MNB.index(max(results_CV_MNB))	
print (results_MNB['mean_test_score'])
print (results_MNB['mean_train_score'])
print "Feature count for max CV accuracy "    +str(feature_count[index_max_accuracy])

with PdfPages('/Users/Dhanush/Desktop/MultinomialNB_Smooth_vs_Accuracy_study.pdf') as pdf:
    pl.plot(feature_count,results_train_MNB,marker='.',markersize = 13.0,linewidth=2, linestyle='-', color='m',label ='Train Score')
    pl.plot(feature_count,results_CV_MNB,marker='.',markersize = 13.0,linewidth=1, linestyle='-', color='b',label ='CV Score')
    pl.ylabel('Classification accuracy',color='r')
    pl.xlabel('Alpha',color='r')
    pl.title('Multinomial NB - Smooth Vs alpha for train using CountVectorizer',color = 'r')
    pl.legend(bbox_to_anchor=(0.69, 0.27), loc=2, borderaxespad=0.)
    pdf.savefig()
    pl.close()
"""
with PdfPages('BernoulliNB_Smooth_vs_Accuracy_study.pdf') as pdf:
    pl.plot([0.000000001,0.00000001,0.0000001,0.0000001],results_BNB['mean_train_score'],marker='.',markersize = 13.0,linewidth=2, linestyle='-', color='m',label ='Train Score')
    pl.plot([0.000000001,0.00000001,0.0000001,0.0000001],results_BNB['mean_test_score'],marker='.',markersize = 13.0,linewidth=1, linestyle='-', color='b',label ='CV Score')
    pl.ylabel('Classification Accuracy',color='r')
    pl.xlabel('Alpha',color='r')
    pl.title('Bernoulli NB - Smooth Vs Alpha using TfidfVectorizer',color = 'r')
    pl.legend(bbox_to_anchor=(0.69, 0.27), loc=2, borderaxespad=0.)
    pdf.savefig()
    pl.close()



print "------- BernoulliNB Naive Bayes-------------------------------------"
index_max_accuracy = results_CV_BNB.index(min(results_CV_BNB))	
print (results_BNB['mean_test_score'])
print (results_BNB['mean_train_score'])
print "Feature count for max CV accuracy "    +str(feature_count[index_max_accuracy])


print "------- GaussianNB Naive Bayes-------------------------------------"
index_max_accuracy = np.argmax(results_GNB['mean_test_score'])
print "CV SCORE"
print (results_GNB['mean_test_score'])
print ""
print (results_GNB['mean_train_score'])
#print "Feature count for max CV accuracy "    +str(feature_count[index_max_accuracy])
print "--------------------------------------------------------------------"
"""
