
# This document is used to study various factors such as no of train documents, alpha, features vs accuracy on Train and cross Validation 
#sets, This uses a 10 fold stratified cv and uses grid search to implement the pipeline. Also plots various graphs.

from newspaper import Article
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
import os
import tldextract
import numpy as np
import re
import nltk
import nltk.tokenize
from scipy.sparse import coo_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn import linear_model
import pickle
from sklearn.externals import joblib
file_loc='/Users/Dhanush/Desktop/Projects/DM_project/DMProj_Data/Data_Final/Train/CODE_'
file_loc1='/Users/Dhanush/Desktop/Projects/DM_project/DMProj_Data/Data_Final/Test/CODE_'
file_loc_out ='/Users/Dhanush/Desktop/Projects/DM_project/DMProj_Data/Data_domain_selected/CODE_'

def data_extract_using_parms(Domain_select,write_output,characters_limit,output_loc,features,file_loc):
	build_data_source =  BuildData(file_loc)
	build_data_source.extract_data_routines()
	data = build_data_source.fetch_train_test_data(Domain_select,write_output,characters_limit,output_loc)
	return data


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
features = 'max'
"""
data=data_extract_using_parms(Domain_select,write_output,characters_limit,file_loc_out,features,file_loc)
total_feature_list = data[0]
total_label = data[1]
print len(total_feature_list)
features = 1500

cv = TfidfVectorizer(input ='total_feature_list',stop_words = {'english'},lowercase=True,analyzer ='word',binary =False,max_features =features)
X = cv.fit_transform(total_feature_list).toarray()
vocab = np.array(cv.get_feature_names())
print "Vocabulary length: " + str(len(vocab))
print "Training size: "     + str(len(total_label)) 
y = (np.array(total_label))
#clf = DecisionTreeClassifier(min_samples_split=2,random_state=0,max_depth =100)
clf = RandomForestClassifier(n_estimators =70)
#clf=linear_model.LogisticRegression(solver="sag",max_iter=1000,C=1500)
#clf1 = svm.SVC(kernel='linear',C =0.1)
clf.fit(X, y)

joblib.dump(clf, 'randomforests.pkl')
joblib.dump(cv, 'tdif.pkl')
"""
clf = joblib.load('randomforests.pkl') 
cv =joblib.load('tdif.pkl') 
print "loaded"
data=data_extract_using_parms(Domain_select,write_output,characters_limit,file_loc_out,features,file_loc1)
total_feature_list = data[0]
X = cv.transform(total_feature_list)
total_label = data[1]
print "Test size: "     + str(len(total_label)) 
y = (np.array(total_label))
out =clf.predict(X)
print "Accuracy:" +str(accuracy_score(y,out))
print "F1 score: "+str(f1_score(y,out, average='macro'))
print "Percision: " + str(precision_score(y,out, average='macro'))
print "Recall: " +str(recall_score(y,out, average='macro'))


#clf1.fit(X, y)
Reddit_articles =[]
red_loc = '/Users/Dhanush/Desktop/test'
for file in os.listdir(red_loc):
    if file.endswith(".csv")or file.endswith(".txt"):
        file_to_read = red_loc + '/' + file 
    	f1 = open(file_to_read,'r')
    	data = f1.read()
    	letters_only = re.sub("[^a-zA-Z]", " ", data) 
        Reddit_articles.append(letters_only)

X = cv.transform(Reddit_articles)
print X
out =clf.predict(X)
print out


