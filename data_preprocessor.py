
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


domain_total_count = {}
domain_top_list = []
total_feature_location = []
total_label = []

file_loc='/Users/Dhanush/Desktop/Projects/DM_project/DMProj_Data/Data_Main_ver1.0/CODE_'
file_loc_out ='/Users/Dhanush/Desktop/Projects/DM_project/DMProj_Data/Data_Domain_selected/CODE_'

class DataPreprocessor(object):
    
    def __init__(self,file_loc,code):
        self._file_loc = file_loc+str(code)
        self._file_loc_out =file_loc_out +str(code)
        self._code = code
        self._domain_count = {}
        self._article_count = 0
        self._domain_selected_article_count = 0
        self._article_to_list = []
        self._label = []
        self._label_array = []
        self._feature = []
        
    # Flag to do domain selection and Write the selected articles
    # returns the list with all news items and labels
    #To do domain selection and write, send as object.FetchData(1,1,2500)
    # length is the minimum left of each articles
    def FetchData(self,flag,towrite,length_limit): 
        if flag == 1:
            if len(domain_top_list) < 1:
                print " You cannot do Domain selection without running count_domain_names Select prior to FetchData"
                return
        try:
            for file in os.listdir(self._file_loc):
                if file.endswith(".txt"):
                    file_to_read = self._file_loc + '/' + file 
                    f = open(file_to_read,'r')
                    num_chars = 0
                    ext = tldextract.extract(f.readline())
                    for line in f : num_chars += len(line)
                    if num_chars < length_limit:
                        continue
                    if (flag == 1 and towrite == 1):
                        if ext.domain in domain_top_list:
                            f1 = open(file_to_read,'r')
                            file_to_write = self._file_loc_out + '/_' + str(self._domain_selected_article_count)+ '.txt'
                            output = open(file_to_write, "w")
                            data = f1.read()
                            letters_only = re.sub("[^a-zA-Z]", " ", data) 
                            output.write(letters_only)
                            self._domain_selected_article_count +=1
                            self._article_to_list.append(letters_only)

                    elif (flag == 1 and towrite == 0):
                        #print ext.domain
                        if ext.domain in domain_top_list:
                            f1 = open(file_to_read,'r')
                            f1.readline()
                            f1.readline()
                            f1.readline()
                            data = f1.read()
                            letters_only = re.sub("[^a-zA-Z]", " ", data) 
                            self._domain_selected_article_count +=1
                            self._article_to_list.append(data)
                    else:
                        f1 = open(file_to_read,'r')
                        f1.readline()
                        f1.readline()
                        f1.readline()
                        data = f1.read()
                        #letters_only = re.sub("[^a-zA-Z]", " ", data) 
                        self._article_to_list.append(data)
        except IOError as err:
            print 'unable to perform Fetch Operation_Codeis: %s' %(self._code)
            print format(err)
            return
        self._label = [self._code for i in range(len(self._article_to_list))]


    def count_domain_names(self):
        
        try:
            for file in os.listdir(self._file_loc):
                if file.endswith(".txt"):
                    file_to_read = self._file_loc + '/' + file
                    f = open(file_to_read,'r') 
                    ext = tldextract.extract(f.readline())
                    self._article_count +=1
                    if ext.domain in self._domain_count:
                        self._domain_count[ext.domain] +=1
                    else:
                        self._domain_count[ext.domain] = 1
        except:
            #print 'unable to open file folder Code_%s' %(self._code)
            return
        #print 'code:' + str(code_dict[i]._code)+ '  article_count:' + str(code_dict[i]._article_count)
        
        
def add_dict(main_dict,temp_dict):
    for domain in temp_dict:
        if domain in main_dict:
            main_dict[domain] +=temp_dict[domain]
        else:
            main_dict[domain] = 1
    return main_dict        

# Required inputs are file location and code for doing Domain analysis
code_dict = {}
total_article_count = 0
for i in range(1,21,1):  
    code = i
    code_dict[i] = DataPreprocessor(file_loc,code)
    code_dict[i].count_domain_names()
    total_article_count += code_dict[i]._article_count
print 'Total Article count across all classes: ' + str(total_article_count)


# collecting domain names accross all classes
#for i in range(1,21,1):
for i in range(8,10,1):
    if len(code_dict[i]._domain_count) == 0:
        continue
    domain_total_count =add_dict(domain_total_count,code_dict[i]._domain_count)

print 'Total Unique Domains' + str(len(domain_total_count))

# selecting most 100 common domains
count = 0
for w in sorted(domain_total_count, key=domain_total_count.get, reverse=True):
    count +=1
    #print  w + " " + str(domain_total_count[w])
    if count < 101 :
        #print  w + " " + str(domain_total_count[w])
        domain_top_list.append(w)

#print domain_top_list   
print ""

# To FetchData from text file, create feature and label list, Do domain selection and write the Domain selcted Data

#for i in range(1,21,1):
for i in range(8,10,1):
    if code_dict[i]._article_count < 1:
        continue
    code_dict[i].FetchData(1,0,2500)
    total_feature_location += code_dict[i]._article_to_list
    total_label   += code_dict[i]._label 

# Combines classes and Vectorize
print "size of training documents" + str(len(total_label))

stop_words = {'english',}
cv = CountVectorizer(input ='total_feature_location',stop_words = {'english'},lowercase=True,analyzer ='word',binary =False)#,max_features =75000)
X = cv.fit_transform(total_feature_location).toarray()
vocab = np.array(cv.get_feature_names())
print vocab[1000:1500]
feature_names = cv.get_feature_names()
y = np.array(total_label)
"""
X_sparse = coo_matrix(X)
X, X_sparse, y = shuffle(X, X_sparse, y, random_state=0)
X = X_sparse.toarray()
"""
X_train, X_test, y_train, y_test = train_test_split(X,y ,test_size=0.2, random_state=5677)


print "Training"
print "Size of Train dataset is :" + str(len(y_train)) + "  " + str(len(X_train))
print "Size of Test dataset is :" + str(len(y_test)) + "  " + str(len(X_test))

alpha = 0.00001
#clf = BernoulliNB(alpha = alpha)
clf = MultinomialNB(alpha = alpha)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print " "
print "train accuracy:" +str(clf.score(X_train,y_train))
print "test accuracy:" +str(clf.score(X_test,y_test))

#print "*"
#print y_pred[1:100]
#print y_test[1:100]
#print ("For MultinomialNB:  " +'alpha=%f ,accuracy = %f' %(alpha, np.mean((y_test-y_pred)==0)))


print ""






