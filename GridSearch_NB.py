
# coding: utf-8

# In[1]:

#This routine reads the raw data of the project which includes 20 classes and over 50000 articles
#and helps select credible and useful articles which can be used for training.


#Do the domain name analysis for each class.

"""
Here we extract the domain names of the source from all the text files. Then we use a python 
dictionary to store the all domain names with their document count. Then we choose the most 
credible ones from each class and write them into Data_processed which can be later used for
testing.
"""

import os
import tldextract
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from scipy.sparse import coo_matrix
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from nltk.tokenize import TreebankWordTokenizer
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import ShuffleSplit
from sklearn.grid_search import GridSearchCV
from sklearn.learning_curve import learning_curve


file_loc='/Users/Dhanush/Desktop/Projects/DM_project/DMProj_Data/Data_Main_ver1.0/CODE_'
file_loc_out ='/Users/Dhanush/Desktop/Projects/DM_project/DMProj_Data/Data_Domain_selected/CODE_'




# Main routine, Create an instance of BuildData and call extract_data_routine

build_data_source =  BuildData(file_loc)
build_data_source.extract_data_routines()
Domain_select = 1
write_output = 0
characters_limit = 4500
output_loc = file_loc_out
data = build_data_source.fetch_train_test_data(Domain_select,write_output,characters_limit,output_loc)
total_feature_list = data[0]
total_label = data[1]
#------------------------------------------------------------------------------
# Vectorize the data
stop_words = {'english',}
cv = CountVectorizer(input ='total_feature_list',stop_words = {'english'},lowercase=True,analyzer ='word',binary =False)#,max_features =75000)
X = cv.fit_transform(total_feature_list).toarray()
vocab = np.array(cv.get_feature_names())
#print vocab[1000:1500]
feature_names = cv.get_feature_names()
y = np.array(total_label)
X_train, X_test, y_train, y_test = train_test_split(X,y ,test_size=0.2, random_state=5677)

print "Training"
print "Size of Train dataset is :" + str(len(y_train)) + "  " + str(len(X_train))
print "Size of Test dataset is :" + str(len(y_test)) + "  " + str(len(X_test))

#--------------------------------------------------------------------------------------------#
#clf = BernoulliNB(alpha = alpha)
clf = MultinomialNB()
param_grid = {'alpha': (1,0.00001) }

"""
# Monte Carlo cross-validation
cv_score = []
train_score = []
for i in range(5):
    X_train, X_CV, y_train, y_CV = train_test_split(X_train,y_train ,test_size=0.15, random_state=5677)
    clf.fit(X_train, y_train)
    #y_pred = clf.predict(X_CV)
    train_score.append(clf.score(X_train,y_train))
    cv_score.append((clf.score(X_CV,y_CV)))
"""
# 5 fold cross validation
cv = ShuffleSplit(X_train.shape[0], n_iter=10, test_size=0.2, random_state=0)
classifier= GridSearchCV(estimator=clf, cv=cv ,param_grid=param_grid)
classifier.fit(X_train, y_train)
print "Grid Search Completed"
#title = 'Learning Curves for MultinomialNB'
#plot_learning_curve(clf, title, X_train, y_train, cv=cv)
#plt.show()
print "Final Classifier score" + str(classifier.score(X_test, y_test))
#print " Train Accuracy: %0.2f " % (sum(train_score)/len(train_score))
#print " CV Accuracy: %0.2f " % (sum(cv_score)/len(cv_score))
#print " Test Accuracy" + str(clf.score(X_test,y_test))
#print "*"
#print y_pred[1:100]
#print y_test[1:100]
#print ("For MultinomialNB:  " +'alpha=%f ,accuracy = %f' %(alpha, np.mean((y_test-y_pred)==0)))


print ""

