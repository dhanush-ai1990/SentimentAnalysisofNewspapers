
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
import gc
from sklearn import svm


file_loc='/Users/Dhanush/Desktop/Projects/DM_project/DMProj_Data/Data_test/CODE_'
file_loc_out ='/Users/Dhanush/Desktop/Projects/DM_project/DMProj_Data/Data_Domain_selected/CODE_'

def data_extract_using_parms(Domain_select,write_output,characters_limit,output_loc,features):
	build_data_source =  BuildData(file_loc)
	build_data_source.extract_data_routines()
	data = build_data_source.fetch_train_test_data(Domain_select,write_output,characters_limit,output_loc)
	return data



def Vectorize_split(total_feature_list,total_label,features,binary):
	stop_words = {'english',}
	if features == 'max':
		cv = CountVectorizer(input ='total_feature_list',stop_words = {'english'},lowercase=True,analyzer ='word',binary =binary)#,non_negative=True)#,max_features =75000)
	else:
		cv = CountVectorizer(input ='total_feature_list',stop_words = {'english'},lowercase=True,analyzer ='word',binary =binary,max_features =features)
	X = cv.fit_transform(total_feature_list)#.toarray()
	vocab = np.array(cv.get_feature_names())
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
	param_grid = {'alpha': [0.00001,0.0000001] }
	# Ten fold Cross Validation
	classifier= GridSearchCV(estimator=clf, cv=10 ,param_grid=param_grid)
	classifier.fit(X_train, y_train)
	return classifier.cv_results_

def MyBernoulliNB(X_train, y_train):
	clf = BernoulliNB()
	param_grid = {'alpha': [0.00001,0.0000001] }
	# Ten fold Cross Validation
	classifier= GridSearchCV(estimator=clf, cv=10 ,param_grid=param_grid)
	classifier.fit(X_train, y_train)
	return classifier.cv_results_

def MyGaussianNB(X_train, y_train):
	clf = GaussianNB()
	param_grid = {}
	# Ten fold Cross Validation
	classifier= GridSearchCV(estimator=clf, cv=10 ,param_grid=param_grid)
	classifier.fit(X_train, y_train)
	return classifier.cv_results_


def Mylinear_svm(X_train, y_train):
	linear_svc = svm.SVC(kernel='linear')#,decision_function_shape ='ovo'/'ovr'
	#param_grid = {'C': np.logspace(-3, 2, 6)}
	param_grid = {'C': [ 0.01, 0.1, 1.0 ,10.0, 100.0]}
	classifier= GridSearchCV(estimator=linear_svc, cv=10 ,param_grid=param_grid)
	y_train= np.array(y_train)
	print y_train
	classifier.fit(X_train, y_train)
	return classifier.cv_results_

def rbf_svm(X_train, y_train):
	rbf_svc = svm.SVC(kernel='rbf')#,max_iter = 10000,cache_size =1024,decision_function_shape ='ovo'/'ovo'
	param_grid = {'C': np.logspace(-3, 2, 6), 'gamma': np.logspace(-3, 2, 6)}
	classifier= GridSearchCV(estimator=rbf_svc, cv=10 ,param_grid=param_grid)
	y_train= np.array(y_train)
	classifier.fit(X_train, y_train)
	return classifier.cv_results_

# main Function
gc.enable()
#Varying Character limit and Document count and study Variance and Biasis
Domain_select = 1
write_output = 0
characters_limit = 0
Document_count = []
character_count =[]
results_LSVM = []
results_train_LSVM =[]
results_CV_LSVM = []
feature_count = []
features = 5000
data=data_extract_using_parms(Domain_select,write_output,characters_limit,file_loc_out,features)
total_feature_list = data[0]
total_label = data[1]
train_test_data=Vectorize_split(total_feature_list,total_label,features,False)
max_features =train_test_data[4]
print max_features
for i in range(1,3,1):
	if features > max_features:
		continue
	print "Training loop: " + str(i)
	feature_count.append(features)
	train_test_data=Vectorize_split(total_feature_list,total_label,features,True)
	print "Features: " + str(train_test_data[4])
	print "Total Docs: " + str((train_test_data[0]).shape[0])
	results_LSVM = Mylinear_svm(train_test_data[0],train_test_data[2])
	results_train_LSVM.append(1 - np.mean(results_LSVM['mean_train_score']))
	results_CV_LSVM.append(1-np.mean(results_LSVM['mean_test_score']))
	features +=5000

with PdfPages('SVM_Linear_kernal_Feature_size.pdf') as pdf:
    pl.plot(feature_count,results_train_LSVM,marker='.',markersize = 13.0,linewidth=2, linestyle='-', color='m',label ='Train Score')
    pl.plot(feature_count,results_CV_LSVM,marker='.',markersize = 13.0,linewidth=1, linestyle='-', color='b',label ='CV Score')
    pl.ylabel('Classification Error',color='r')
    pl.xlabel('Number of features',color='r')
    pl.title('SVM_linear_kernal - Error Vs # of features for training',color = 'r')
    pl.legend(bbox_to_anchor=(0.69, 0.27), loc=2, borderaxespad=0.)
    pdf.savefig()
    pl.close()
"""
with PdfPages('MultinomialNB_Character-LIMIT_study.pdf') as pdf:
    pl.plot(character_count,results_train,marker='.',markersize = 13.0,linewidth=2, linestyle='-', color='m',label ='Train Score')
    pl.plot(character_count,results_CV,marker='.',markersize = 13.0,linewidth=1, linestyle='-', color='b',label ='CV Score')
    pl.ylabel('Classification Error',color='r')
    pl.xlabel('Number Of minimum characters per document',color='r')
    pl.title('Multinomial NB - Error Vs # of Character per doc for train with domain sel',color = 'r')
    pl.legend(bbox_to_anchor=(0.69, 0.27), loc=2, borderaxespad=0.)
    pdf.savefig()
    pl.close()
"""



print "------- Linear SVM-------------------------------------"
#index_max_accuracy = results_CV_LSVM.index(min(results_CV_LSVM))	
print "Maximum CV accuracy : " + str(1- min(results_CV_LSVM))
print ""
print "Train scores: " + (results_LSVM['mean_train_score'])
print "CV scores: " + (results_LSVM['mean_test_score'])
#print "character_count for max CV accuracy : " + str(character_count[index_max_accuracy])
#print "Document_count for max CV accuracy : " + str (Document_count[index_max_accuracy])
#print "Feature count for max CV accuracy "    +str(feature_count[index_max_accuracy])
print "--------------------------------------------------------------------"







