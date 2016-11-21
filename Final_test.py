
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

file_loc='/Users/Dhanush/Desktop/Projects/DM_project/DMProj_Data/Data_train/CODE_'
file_loc_out ='/Users/Dhanush/Desktop/Projects/DM_project/DMProj_Data/Data_domain_selected/CODE_'

def data_extract_using_parms(Domain_select,write_output,characters_limit,output_loc,features):
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
data=data_extract_using_parms(Domain_select,write_output,characters_limit,file_loc_out,features)
total_feature_list = data[0]
total_label = data[1]
print len(total_feature_list)
features = 5000
cv = TfidfVectorizer(input ='total_feature_list',stop_words = {'english'},lowercase=True,analyzer ='word',binary =False,max_features =features)
X = cv.fit_transform(total_feature_list)
vocab = np.array(cv.get_feature_names())
print len(vocab)
y = (np.array(total_label))
clf = MultinomialNB(alpha = 0.00001)
#clf1 = svm.SVC(kernel='linear',C =0.1)
clf.fit(X, y)
#clf1.fit(X, y)
Reddit_articles =[]
red_loc = '/Users/Dhanush/Desktop/Projects/DM_project/DMProj_Data/Data_reddit/CODE_1'
for file in os.listdir(red_loc):
    if file.endswith(".csv")or file.endswith(".txt"):
        file_to_read = red_loc + '/' + file 
    	f1 = open(file_to_read,'r')
    	data = f1.read()
    	letters_only = re.sub("[^a-zA-Z]", " ", data) 
        Reddit_articles.append(letters_only)

article=Article("http://www.mirror.co.uk/news/uk-news/detectives-launch-murder-investigation-after-9282103")
article.download()
article.parse()
text=article.text.encode("utf-8")
text =[r"A woman was raped multiple times by a masked man. She suffered horrible injuries. police investigating the case"]
X = cv.transform(Reddit_articles)
out =clf.predict(X)
#out1 =clf1.fit(X, y)
"""
check = []

for i in range(len(out)):
	if out[i] = out1[i]:
		check.append['M']
	else:
		check.append['N']
"""




