

from sklearn import svm, datasets
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
iris = datasets.load_iris()
#parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
parameters = {'alpha': (1,0.00001) }
#svr = svm.SVC()
svr = MultinomialNB()
clf = GridSearchCV(estimator =svr, param_grid=parameters,cv = 10)
clf.fit(iris.data, iris.target)
print (sorted(clf.cv_results_))