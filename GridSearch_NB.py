
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


file_loc='/Users/Dhanush/Desktop/Projects/DM_project/DMProj_Data/Data_Main_ver1.0/CODE_'
file_loc_out ='/Users/Dhanush/Desktop/Projects/DM_project/DMProj_Data/Data_Domain_selected/CODE_'

def data_extract_using_parms(Domain_select,write_output,characters_limit,output_loc):
	build_data_source =  BuildData(file_loc)
	build_data_source.extract_data_routines()
	data = build_data_source.fetch_train_test_data(Domain_select,write_output,characters_limit,output_loc)
	total_feature_list = data[0]
	total_label = data[1]
	return Vectorize_split(total_feature_list,total_label)


def Vectorize_split(total_feature_list,total_label):
	stop_words = {'english',}
	cv = CountVectorizer(input ='total_feature_list',stop_words = {'english'},lowercase=True,analyzer ='word',binary =False)#,max_features =75000)
	X = cv.fit_transform(total_feature_list).toarray()
	vocab = np.array(cv.get_feature_names())
	feature_names = cv.get_feature_names()
	y = np.array(total_label)
	train_test_data = [i for i in range(4)]
	#X_train, X_test, y_train, y_test = train_test_split(X,y ,test_size=0.2, random_state=5677)
	train_test_data[0],train_test_data[1],train_test_data[2],train_test_data[3] = train_test_split(X,y ,test_size=0.2, random_state=5677)
	return train_test_data
# Main routine, Create an instance of BuildData and call extract_data_routine

def MyMultiNomialNB(X_train, y_train):
	clf = MultinomialNB()
	param_grid = {'alpha': [0.00001,1] }
	# Ten fold Cross Validation
	classifier= GridSearchCV(estimator=clf, cv=10 ,param_grid=param_grid)
	classifier.fit(X_train, y_train)
	return classifier.cv_results_


# main Function

#Varying Character limit and Document count and study Variance and Biasis
Domain_select = 1
write_output = 0
characters_limit = 6000
results_CV = []
results_train = []
Document_count = []
character_count = []

for i in range(1,3,1):
	print "Training loop: " + str(i)
	train_test_data=data_extract_using_parms(Domain_select,write_output,characters_limit,file_loc_out)
	Document_count.append(len(train_test_data[0]))
	character_count.append(characters_limit) 
	print "Vectorise Completed for :" +str(i)
	results = MyMultiNomialNB(train_test_data[0],train_test_data[2])
	results_train.append(np.mean(results['mean_train_score']))
	results_CV.append(np.mean(results['mean_test_score']))
	characters_limit +=500
	
print Document_count
print results_train
print results_CV
print character_count

with PdfPages('MultinomialNB_TRAIN SIZE_study.pdf') as pdf:
	pl.ylim([0,1.0])
    pl.plot(Document_count,results_train,marker='.',markersize = 13.0,linewidth=2, linestyle='-', color='m',label ='Train Score')
    pl.plot(Document_count,results_CV,marker='.',markersize = 13.0,linewidth=1, linestyle='-', color='b',label ='CV Score')
    pl.ylabel('Accuracy',color='r')
    pl.xlabel('Number Of Documents for Training',color='r')
    pl.title('Multinomial NB - Accuracy Vs no of Documents for train',color = 'r')
    pl.legend(bbox_to_anchor=(0.69, 0.27), loc=2, borderaxespad=0.)
    pdf.savefig()
    pl.close()
with PdfPages('MultinomialNB_Character-LIMIT_study.pdf') as pdf:
	pl.ylim([0,1.0])
    pl.xlim([0,8000])
    pl.plot(character_count,results_train,marker='.',markersize = 13.0,linewidth=2, linestyle='-', color='m',label ='Train Score')
    pl.plot(character_count,results_CV,marker='.',markersize = 13.0,linewidth=1, linestyle='-', color='b',label ='CV Score')
    pl.ylabel('Accuracy',color='r')
    pl.xlabel('Number Of Documents for Training',color='r')
    pl.title('Multinomial NB - Accuracy Vs no of Character per doc for train',color = 'r')
    pl.legend(bbox_to_anchor=(0.69, 0.27), loc=2, borderaxespad=0.)
    pdf.savefig()
    pl.close()
print ""

