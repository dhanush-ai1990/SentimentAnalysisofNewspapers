
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
from sklearn.feature_extraction.text import CountVectorizer

domain_total_count = {}
domain_top_list = []

file_loc='/Users/Dhanush/Desktop/Projects/DM_project/DMProj_Data/Data_Main_ver1.0/CODE_'
file_loc_out ='/Users/Dhanush/Desktop/Projects/DM_project/DMProj_Data/Data_Domain_selected/CODE_'

class DomainSelector(object):
    
    def __init(self):
        pass
        
    def DomainSelect(self):
        try:
            for file in os.listdir(self._file_loc):
                if file.endswith(".txt"):
                    file_to_read = self._file_loc + '/' + file 
                    f = open(file_to_read,'r') 
                    ext = tldextract.extract(f.readline())
                    if ext.domain in domain_top_count:
                        f1 = open(file_to_read,'r')
                        file_to_write = self._file_loc_out + '/_' + str(self._domain_selected_article_count)+ '.txt'
                        output = open(file_to_write, "w")
                        output.write(f1.read())
                        self._domain_selected_article_count +=1               
        except:
            print 'unable to open file folder Code_%s' %(self._code)
        print 'code:' + str(code_dict[i]._code)+ '  Selected_article_count:' + str(code_dict[i]._article_count)
        print ""   
        return

    # Flag to do domain selection and Write the selected articles
    # returns the list with all news items and labels
    #To do domain selection and write, send as object.FetchData(1,1,2500)
    # length is the minimum left of each articles
    def FetchData(self,flag,towrite,length_limit): 
        if flag == 1
        try:
            for file in os.listdir(self._file_loc):
                if file.endswith(".txt"):
                    file_to_read = self._file_loc + '/' + file 
                    f = open(file_to_read,'r')

                    if len(f) < length_limit:
                        continue

                    ext = tldextract.extract(f.readline())

                    if (flag == 1 and towrite == 1):
                        if ext.domain in domain_top_count:
                            f1 = open(file_to_read,'r')
                            file_to_write = self._file_loc_out + '/_' + str(self._domain_selected_article_count)+ '.txt'
                            output = open(file_to_write, "w")
                            data = f1.read()
                            output.write(data)
                            self._domain_selected_article_count +=1
                            self._article_to_list.append(data)

                    elif (flag == 1 and towrite == 0):
                        if ext.domain in domain_top_count:
                            f1 = open(file_to_read,'r')
                            data = f1.read()
                            self._domain_selected_article_count +=1
                            self._article_to_list.append(data)
                    else:
                        f1 = open(file_to_read,'r')
                        data = f1.read()
                        self._article_to_list.append([data])
        except:
            print 'unable to perform Fetch Operation_%s' %(self._code)
            print ""
            return
        self._label = [code for i in range(len(self._article_to_list.append())])


class DataPreprocessor(DomainSelector):
    
    def __init__(self,file_loc,code):
        self._file_loc = file_loc+str(code)
        self._file_loc_out =file_loc_out +str(code)
        self._code = code
        self._domain_count = {}
        self._article_count = 0
        self._domain_selected_article_count = 0
        self._article_to_list = []
        self._label = []
        self._feature = []

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
            print 'unable to open file folder Code_%s' %(self._code)
            print ""
            return
        print 'code:' + str(code_dict[i]._code)+ '  article_count:' + str(code_dict[i]._article_count)
        print ""
        
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
for i in range(1,21,1):
    if len(code_dict[i]._domain_count) == 0:
        continue
    
    domain_total_count =add_dict(domain_total_count,code_dict[i]._domain_count)

# selecting most 100 common domains
count = 0
for w in sorted(domain_total_count, key=domain_total_count.get, reverse=True):
    count +=1
    print  w + " " + str(domain_total_count[w])
    if count < 101 :
        #print  w + " " + str(domain_total_count[w])
        domain_top_count.list(w)
print domain_top_list   
print ""

# To select the news based on Domains, run below code.
#for i in range(1,21,1):
    #code_dict[i].DomainSelect()





