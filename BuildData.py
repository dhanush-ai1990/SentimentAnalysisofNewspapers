import os
import tldextract
import numpy as np
import re
from scipy.sparse import coo_matrix
#file_loc='/Users/Dhanush/Desktop/Projects/DM_project/DMProj_Data/Data_Main_ver1.0/CODE_'
#file_loc_out ='/Users/Dhanush/Desktop/Projects/DM_project/DMProj_Data/Data_Domain_selected/CODE_'


class BuildData(object):
    
    def __init__(self,file_loc):
        self._file_loc = file_loc
        self.code_dict = {}
        self.total_article_count = 0
        self.domain_total_count = {}
        self.domain_top_list = []
        self.total_feature_location = []
        self.total_label = []

    def add_dict(self,main_dict,temp_dict):
        for domain in temp_dict:
            if domain in main_dict:
                main_dict[domain] +=temp_dict[domain]
            else:
                main_dict[domain] = 1
        return main_dict   

    def extract_data_routines(self):
        #Required inputs are file location and code for doing Domain analysis
        
        for i in range(1,21,1):  
            code = i
            self.code_dict[i] = DataPreprocessor(file_loc,code)
            self.code_dict[i].count_domain_names()
            self.total_article_count += self.code_dict[i]._article_count
        print 'Total Article count across all classes: ' + str(self.total_article_count)

        # collecting domain names accross all classes
        for i in range(1,21,1):
        #for i in range(8,10,1):
            if len(self.code_dict[i]._domain_count) == 0:
                continue
            self.domain_total_count =self.add_dict(self.domain_total_count,self.code_dict[i]._domain_count)

        print 'Total Unique Domains: ' + str(len(self.domain_total_count))

        #print domain_top_list  
        count = 0
        for w in sorted(self.domain_total_count, key=self.domain_total_count.get, reverse=True):
            count +=1
        #print  w + " " + str(domain_total_count[w])
            if count < 101 :
                #print  w + " " + str(domain_total_count[w])
                self.domain_top_list.append(w)
        print ""
    
    def fetch_train_test_data(self,domain_select,write_output,char_limit,output_loc): 
        data = []
        for i in range(1,21,1):
            if self.code_dict[i]._article_count < 1:
                continue
            self.code_dict[i].FetchData(domain_select,write_output,char_limit,output_loc,self.domain_top_list)
            self.total_feature_location += self.code_dict[i]._article_to_list
            self.total_label   += self.code_dict[i]._label 
        data.append(self.total_feature_location)
        data.append(self.total_label)
        print "size of training documents: " + str(len(self.total_label))
        return   data  
        
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
    def FetchData(self,flag,towrite,length_limit,file_loc_out,domain_top_list):
        self._file_loc_out = file_loc_out +str(self._code)
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