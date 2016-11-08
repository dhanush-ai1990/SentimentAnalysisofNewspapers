
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
	classifier= GridSearchCV(estimator=clf, cv=3 ,param_grid=param_grid)
	classifier.fit(X_train, y_train)
	return classifier.cv_results_

def MyRandomForest(X_train, y_train):
	clf = RandomForestClassifier()
	param_grid = {'n_estimators': [10,20,30,50,70,100]}
	classifier= GridSearchCV(estimator=clf, cv=3 ,param_grid=param_grid)
	classifier.fit(X_train, y_train)
	return classifier.cv_results_

def MyDecisionTree(X_train, y_train):
	clf = DecisionTreeClassifier(min_samples_split=2,random_state=0)
	param_grid = {'max_depth': [10,25,50,75,100]}
	classifier= GridSearchCV(estimator=clf, cv=3 ,param_grid=param_grid)
	classifier.fit(X_train, y_train)
	return classifier.cv_results_

def MyExtraTreeClassifier(X_train, y_train):
	clf = ExtraTreesClassifier(min_samples_split=2, random_state=0)
	param_grid ={'n_estimators': [10,20,30,40,50,75,100]}
	#param_grid = {'max_depth': [10,25,50,75,100]}
	classifier= GridSearchCV(estimator=clf, cv=3 ,param_grid=param_grid)
	classifier.fit(X_train, y_train)
	return classifier.cv_results_

def MyAdaBoostClassifier(X_train, y_train):
	clf = AdaBoostClassifier()
	param_grid = {'n_estimators': [10,20,30,50,70,100]}
	classifier= GridSearchCV(estimator=clf, cv=3 ,param_grid=param_grid)
	classifier.fit(X_train, y_train)
	return classifier.cv_results_

# main Function
gc.enable()

#Execution Step for Ensemble Method Classifiers
# Use count Vectorizer but with very few samples and number of features < 200.
#Steps For implementation.
#1) Find the Number of features and keep it fixed for each classifier
#2) varying Tuning parameters like Depth and no of estimates
#3) Plot graph for each case
#4) Extremely random classifier needs both parameters to be varied
#5) Do not give too many inputs or features since these will overfit easily.
Domain_select = 1
write_output = 0
characters_limit = 10
Document_count = []
results_CV_DTC = []
results_train_DTC = []
results_CV_ETC = []
results_train_ETC = []
results_CV_RFC = []
results_train_RFC = []
results_CV_ABC = []
results_train_ABC = []
feature_count = []
features = 'max'
data=data_extract_using_parms(Domain_select,write_output,characters_limit,file_loc_out,features)
total_feature_list = data[0]
total_label = data[1]
features = 100

for i in range(1,21,1):
	print i
	train_test_data=Vectorize_split(total_feature_list,total_label,features,False)
	"""
	results_DTC = MyDecisionTree(train_test_data[0],train_test_data[2])
	results_train_DTC.append(1 - np.mean(results_DTC['mean_train_score']))
	results_CV_DTC.append(1-np.mean(results_DTC['mean_test_score']))
	results_ETC = MyExtraTreeClassifier(train_test_data[0],train_test_data[2])
	results_train_ETC.append(1 - np.mean(results_ETC['mean_train_score']))
	results_CV_ETC.append(1-np.mean(results_ETC['mean_test_score']))
	results_RFC = MyRandomForest(train_test_data[0],train_test_data[2])
	results_train_RFC.append(1 - np.mean(results_RFC['mean_train_score']))
	results_CV_RFC.append(1-np.mean(results_RFC['mean_test_score']))
	"""
	results_ABC = MyAdaBoostClassifier(train_test_data[0],train_test_data[2])
	results_train_ABC.append(1 - np.mean(results_ABC['mean_train_score']))
	results_CV_ABC.append(1-np.mean(results_ABC['mean_test_score']))
	feature_count.append(train_test_data[4])
	Document_count.append((train_test_data[0]).shape[0])
	features +=10
"""
with PdfPages('Decision_Tree_Feature_size_vs_Error_study.pdf') as pdf:
    pl.plot(feature_count,results_train_DTC,marker='.',markersize = 13.0,linewidth=2, linestyle='-', color='m',label ='Train Score')
    pl.plot(feature_count,results_CV_DTC,marker='.',markersize = 13.0,linewidth=1, linestyle='-', color='b',label ='CV Score')
    pl.ylabel('Classification Error',color='r')
    pl.xlabel('Number Of features for Training',color='r')
    pl.title('Decision Tree - Error Vs # of features for train using CountVectorizer',color = 'r')
    pl.legend(bbox_to_anchor=(0.69, 0.27), loc=2, borderaxespad=0.)
    pdf.savefig()
    pl.close()

with PdfPages('Random_forests_Feature_size_vs_Error_study.pdf') as pdf:
    pl.plot(feature_count,results_train_RFC,marker='.',markersize = 13.0,linewidth=2, linestyle='-', color='m',label ='Train Score')
    pl.plot(feature_count,results_CV_RFC,marker='.',markersize = 13.0,linewidth=1, linestyle='-', color='b',label ='CV Score')
    pl.ylabel('Classification Error',color='r')
    pl.xlabel('Number Of features for Training',color='r')
    pl.title('Random_forests - Error Vs # of features for train',color = 'r')
    pl.legend(bbox_to_anchor=(0.69, 0.27), loc=2, borderaxespad=0.)
    pdf.savefig()
    pl.close()

with PdfPages('extremely Random Tree_Feature_size_vs_Error_study.pdf') as pdf:
    pl.plot(feature_count,results_train_ETC,marker='.',markersize = 13.0,linewidth=2, linestyle='-', color='m',label ='Train Score')
    pl.plot(feature_count,results_CV_ETC,marker='.',markersize = 13.0,linewidth=1, linestyle='-', color='b',label ='CV Score')
    pl.ylabel('Classification Error',color='r')
    pl.xlabel('Number Of features for Training',color='r')
    pl.title('Extree Random_Tree - Error Vs # of features for train',color = 'r')
    pl.legend(bbox_to_anchor=(0.69, 0.27), loc=2, borderaxespad=0.)
    pdf.savefig()
    pl.close()
"""
with PdfPages('AdaBoost_Feature_size_vs_Error_study.pdf') as pdf:
    pl.plot(feature_count,results_train_ABC,marker='.',markersize = 13.0,linewidth=2, linestyle='-', color='m',label ='Train Score')
    pl.plot(feature_count,results_CV_ABC,marker='.',markersize = 13.0,linewidth=1, linestyle='-', color='b',label ='CV Score')
    pl.ylabel('Classification Error',color='r')
    pl.xlabel('Number Of features for Training',color='r')
    pl.title('Adaboost - Error Vs # of features for train using CountVectorizer',color = 'r')
    pl.legend(bbox_to_anchor=(0.69, 0.27), loc=2, borderaxespad=0.)
    pdf.savefig()
    pl.close()

"""
print "------- Decision Tree Classifier-------------------------------------"
index_max_accuracy = results_CV_DTC.index(min(results_CV_DTC))	
print "Maximum CV accuracy : " + str(1- min(results_CV_DTC))
print "Maximum Train accuracy: " + str(1-min(results_train_DTC))
print "Feature count for max CV accuracy "    +str(feature_count[index_max_accuracy])
print "------- RandomForestClassifier-------------------------------------"
index_max_accuracy = results_CV_RFC.index(min(results_CV_RFC))	
print "Maximum CV accuracy : " + str(1- min(results_CV_RFC))
print "Maximum Train accuracy: " + str(1-min(results_train_RFC))
print "Feature count for max CV accuracy "    +str(feature_count[index_max_accuracy])
print "------- Extemely Random Tree Classifier-------------------------------------"
index_max_accuracy = results_CV_ETC.index(min(results_CV_ETC))	
print "Maximum CV accuracy : " + str(1- min(results_CV_ETC))
print "Maximum Train accuracy: " + str(1-min(results_train_ETC))
print "Feature count for max CV accuracy "    +str(feature_count[index_max_accuracy])
"""
print "------- AdaBoostClassifier-------------------------------------"
index_max_accuracy = results_CV_ABC.index(min(results_CV_ABC))	
print "Maximum CV accuracy : " + str(1- min(results_CV_ABC))
print "Maximum Train accuracy: " + str(1-min(results_train_ABC))
print "Feature count for max CV accuracy "    +str(feature_count[index_max_accuracy])

print str(max(Document_count))


