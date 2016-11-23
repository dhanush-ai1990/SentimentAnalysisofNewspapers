
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
from sklearn import linear_model

file_loc='/Users/Dhanush/Desktop/Projects/DM_project/DMProj_Data/Data_Final/Train/CODE_'
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
		cv = TfidfVectorizer(input ='total_feature_list',lowercase=True,binary =binary,norm='l2',sublinear_tf =True,stop_words = {'english'},max_features=features)
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

def MyRandomForest(X_train, y_train):
	clf = RandomForestClassifier()
	param_grid = {'n_estimators': [70]}
	classifier= GridSearchCV(estimator=clf, cv=3 ,param_grid=param_grid)
	classifier.fit(X_train, y_train)
	return classifier.cv_results_

def MyDecisionTree(X_train, y_train):
	clf = DecisionTreeClassifier(min_samples_split=2,random_state=0)
	param_grid = {'max_depth': [1,5,10,25,50,75,100,500,1000,2000]}
	classifier= GridSearchCV(estimator=clf, cv=3 ,param_grid=param_grid)
	classifier.fit(X_train, y_train)
	return classifier.cv_results_

def MyExtraTreeClassifier(X_train, y_train):
	clf = ExtraTreesClassifier(min_samples_split=2, random_state=0,max_depth = 30)
	param_grid = {'n_estimators': [10,20,30,40,50,60,70,80,90,100]}
	#param_grid = {'max_depth': [1,5,10,25,50,75,100,500,1000,2000]}
	classifier= GridSearchCV(estimator=clf, cv=3 ,param_grid=param_grid)
	classifier.fit(X_train, y_train)
	return classifier.cv_results_

def MyAdaBoostClassifier(X_train, y_train):
	#clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=10),learning_rate=0.5)
	clf = AdaBoostClassifier(linear_model.LogisticRegression(solver="sag",max_iter=1000,C=15))
	param_grid = {'learning_rate' : [0.1]}
	classifier= GridSearchCV(estimator=clf, cv=3,param_grid=param_grid)
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
# main Function
gc.enable()
#Varying Character limit and Document count and study Variance and Biasis
Domain_select = 0
write_output = 0
characters_limit = 5000
Document_count = []
character_count =[]
results_CV_MNB = []
results_train_MNB = []
results_CV_GNB = []
results_train_GNB = []
results_CV_BNB = []
results_train_BNB = []
feature_count = []
min_df_count =[]
max_df_count =[]
depth_count =[]
features = 'max'
data=data_extract_using_parms(Domain_select,write_output,characters_limit,file_loc_out,features)
total_feature_list = data[0]
print len(total_feature_list)
total_label = data[1]
#train_test_data=Vectorize_split(total_feature_list,total_label,features,False)
#max_features =train_test_data[4]
#print max_features
features = 1000
#depth = 10
for i in range(1,2,1):
	print i

	
	train_test_data=Vectorize_split(total_feature_list,total_label,features,False)
	print train_test_data[4]
	results_MNB = MyAdaBoostClassifier(train_test_data[0],train_test_data[2])
	results_train_MNB.append(np.mean(results_MNB['mean_train_score']))
	results_CV_MNB.append(np.mean(results_MNB['mean_test_score']))
	
	#Document_count.append((train_test_data[0]).shape[0])
	#character_count.append(characters_limit) 
	feature_count.append(train_test_data[4])
	#features+=100
	#depth_count.append(depth)
	#depth +=5
	


print "-------Extra Tree Classifier-------------------------------------"
print "mean CV scores"
print results_MNB['mean_test_score']
print "mean train scores"
print results_MNB['mean_train_score']
print 'estimators:'


"""
with PdfPages('/Users/Dhanush/Desktop/RandomForests_Estimators_vs_Accuracy_Count.pdf') as pdf:
    pl.plot(max_df_count,results_train_MNB,marker='.',markersize = 13.0,linewidth=2, linestyle='-', color='m',label ='Train Score')
    pl.plot(max_df_count,results_CV_MNB,marker='.',markersize = 13.0,linewidth=1, linestyle='-', color='b',label ='CV Score')
    pl.ylabel('Classification accuracy',color='r')
    pl.xlabel('max_df',color='r')
    pl.title('RandomForest estimators Vs Accuracy using TDIFVectorizer',color = 'r')
    pl.legend(bbox_to_anchor=(0.69, 0.57), loc=2, borderaxespad=0.)
    pdf.savefig()
    pl.close()

"""
