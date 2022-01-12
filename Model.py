#Load in our librabries
from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
print(type(twenty_train))
print("Thu the tham said: Data has been loaded")
####################
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# Some popular evaluation metrics for classification models
from sklearn.metrics import accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')
####################
from tkinter import Tk
from tkinter import filedialog

#Select file
Tk().withdraw()
openfilename = filedialog.askopenfilename(
title='slect_file_csv', initialdir='./')
data = pd.read_csv(openfilename)
x = data.iloc[:, 0]
y = data.iloc[:, 1]
print("Thu the tham said: The file has been read")

    # Divide data into train-test-sets
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state = 42)
print("Size of x_train, y_train, x_test, y_test")
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    #Paste our data in sklearn-dataset
twenty_train.data = x_train
twenty_test.data = x_test
twenty_train.target = y_train
twenty_test.target = y_test

print("Thu the tham said: Size of train set and test set")
print(len(twenty_train.data)) 
print(len(twenty_test.data))


# Preprocessing: Bag of Words and Tf-idf (sklearn)
# Extracting features from text files
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)

# TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print("Thu the tham said: Bag of words has been created with the size of:")
print(X_train_tfidf.shape)

# fit a Naive Bayes model to the data
from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB


# Using Scikit-learn Pipelines to combine preprocessing and training model steps
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import ShuffleSplit

from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
print("Thu the tham said: Trainihg Naive bayes, without using stopword and stemming")
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)
print('Thu the tham said: Done training Naive bayes in', train_time, 'seconds.')
print("Thu the tham said: Diagram is being created")
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.5, 1.2, 5)):
  
 
  
  plt.figure()
  plt.title(title)
  if ylim is not None:
    plt.ylim(*ylim)
  plt.xlabel("Training examples")
  plt.ylabel("Score")
  train_sizes, train_scores, test_scores = learning_curve(
      estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
  train_scores_mean = np.mean(train_scores, axis=1)
  train_scores_std = np.std(train_scores, axis=1)
  test_scores_mean = np.mean(test_scores, axis=1)
  test_scores_std = np.std(test_scores, axis=1)
  plt.grid()

  plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
  plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
  plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
  plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

  plt.legend(loc="best")
  return plt

estimator = MultinomialNB()
title = "Learning Curves (Naive Bayes)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
print("Thu the tham said: Turn it off to continue!")
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
X, y = X_train_tfidf, twenty_train.target
plot_learning_curve(estimator, title, X, y, ylim=(0.0, 1.01), cv=cv, n_jobs=8)
plt.show()

# Improve the previous models by removing stopwords and regular words, and using GridSearchCV

# NLTK
# Removing stop words
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()), 
                     ('clf', MultinomialNB())])
# Stemming Code
import nltk
nltk.download('stopwords')
print('steming the corpus... Please wait...')
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english", ignore_stopwords=True)
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])
    
stemmed_count_vect = StemmedCountVectorizer(stop_words='english')

text_mnb_stemmed = Pipeline([('vect', stemmed_count_vect), ('tfidf', TfidfTransformer()), 
                             ('mnb', MultinomialNB(fit_prior=False))])


#Training NB, calculating its performance                       
print("Thu the tham said: Training Naive bayes, using stopword and stemming")
start_time = time.time()
text_mnb_stemmed = text_mnb_stemmed.fit(twenty_train.data, twenty_train.target)
train_time = time.time() - start_time
print('Thu the tham said: Done training Naive bayes, using stopword and stemming, in', train_time, 'seconds.')
predicted_mnb_stemmed = text_mnb_stemmed.predict(twenty_test.data)

np.mean(predicted_mnb_stemmed == twenty_test.target)




#Training Decision Tree and its performance
    # Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
    # Import accuracy_score
from sklearn.metrics import accuracy_score
start_time = time.time()
text_clf_dt = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-dt', DecisionTreeClassifier(random_state=42))])
print("Thu the tham said: Training Decision Tree, using stopword and stemming")
text_clf_dt = text_clf_dt.fit(twenty_train.data, twenty_train.target)
train_time = time.time() - start_time
print('Thu the tham said: Done training Decision Tree, using stopword and  stemming, in', train_time, 'seconds.')
predicted_dt = text_clf_dt.predict(twenty_test.data)
np.mean(predicted_dt == twenty_test.target)




#Training Random Forest and calculating its performance
    # Import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import RandomForestClassifier
text_clf_rf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-dt', RandomForestClassifier(random_state=42))])
print("Thu the tham said: Training Random Forest, using stopword and stemming")
start_time = time.time()
text_clf_rf = text_clf_rf.fit(twenty_train.data, twenty_train.target)
train_time = time.time() - start_time
print('Thu the tham said: Done training Random Forest, using stopword and  stemming, in', train_time, 'seconds.')
predicted_rf = text_clf_rf.predict(twenty_test.data)
np.mean(predicted_rf == twenty_test.target)





#Training KNN and calculating its performance
from sklearn.neighbors import KNeighborsClassifier
text_clf_KNN = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-svm', KNeighborsClassifier(n_neighbors=5))])
print("Thu the tham said: Training KNN, using stopword and stemming")
start_time = time.time()
text_clf_KNN = text_clf_KNN.fit(twenty_train.data, twenty_train.target)
train_time = time.time() - start_time
print('Thu the tham said: Done training KNN, using stopword and  stemming, in', train_time, 'seconds.')
predicted_KNN = text_clf_KNN.predict(twenty_test.data)
np.mean(predicted_KNN == twenty_test.target)




# Training Support Vector Machines - SVM and calculating its performance
from sklearn.linear_model import SGDClassifier
text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter_no_change=5, 
                                                   random_state=42))])
print("Thu the tham said: Training SVM, using stopword and stemming")
start_time = time.time()
text_clf_svm = text_clf_svm.fit(twenty_train.data, twenty_train.target)
train_time = time.time() - start_time
print('Thu the tham said: Done training KNN, using stopword and  stemming, in', train_time, 'seconds.')
predicted_svm = text_clf_svm.predict(twenty_test.data)
np.mean(predicted_svm == twenty_test.target)


#Evaluating and confusion matrix
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix

#Confusion matrix of SVM
plot_confusion_matrix(text_clf_svm, title = 'Support Vector Machine', twenty_test.data, twenty_test.target)  


#Confusion matrix of Decision Tree
plot_confusion_matrix(text_clf_dt, title = 'Decision Tree', twenty_test.data, twenty_test.target)  


#Confusion matrix of RandomForest
plot_confusion_matrix(text_clf_rf, title = 'Random Forest', twenty_test.data, twenty_test.target)  


#Confusion matrix of Naive Bayes
plot_confusion_matrix(text_mnb_stemmed, title = 'Random Forest', twenty_test.data, twenty_test.target)  


#Confusion matrix of KNN
plot_confusion_matrix(text_clf_KNN, twenty_test.data, twenty_test.target)  
plt.show()


#Evaluation matrix of NB
print("Thu the tham said: Results of NB + Stopwords")
# make predictions
expected = twenty_test.target
#print(expected)
predicted = text_mnb_stemmed.predict(twenty_test.data)
#print(predicted)
# summarize the fit of  model
print(metrics.classification_report(expected, predicted))


#Evaluation matrix of Decision Tree
print("Thu the tham said: Results of Decision Tree + Stopwords")
# make predictions
expected = twenty_test.target
#rint(expected)
predicted = text_clf_dt.predict(twenty_test.data)
#print(predicted)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))


#Evaluation matrix of Random Forest
print("Thu the tham said: Results of Random Forest + Stopwords")
# make predictions
expected = twenty_test.target
#rint(expected)
predicted = text_clf_rf.predict(twenty_test.data)
#print(predicted)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))


#Evaluation matrix of KNN
print("Thu the tham said: Results of KNN + Stopwords")
# make predictions
expected = twenty_test.target
#rint(expected)
predicted = text_clf_KNN.predict(twenty_test.data)
#print(predicted)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))


#Evaluation matrix of SVM
print("Thu the tham said: Results of SVM + Stopwords")
# make predictions
expected = twenty_test.target
#rint(expected)
predicted = text_clf_svm.predict(twenty_test.data)
#print(predicted)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))

#Creating Learning Curves Diagram of NB
estimator = MultinomialNB()
title = "Learning Curves (Naive Bayes)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
print("Thu the tham said: Creating diagram, Naive Bayes")
X, y = X_train_tfidf, twenty_train.target
plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.1), cv=cv, n_jobs=8)


#Creating Learning Curves Diagram of Decision Tree
estimator = DecisionTreeClassifier()
title = "Learning Curves (Decision Tree)"
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
print("Thu the tham said: Creating diagram, Decision Tree")
plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.1), cv=cv, n_jobs=8)

#Creating Learning Curves Diagram of RandomForest
estimator = RandomForestClassifier()
title = "Learning Curves (RandomForest)"
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
print("Thu the tham said: Creating diagram, RandomForest")
plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.1), cv=cv, n_jobs=8)

#Creating Learning Curves Diagram of HNN
estimator = KNeighborsClassifier()
title = "Learning Curves (KNN)"
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
print("Thu the tham said: Creating diagram, KNN")
plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.1), cv=cv, n_jobs=8)

#Creating Learning Curves Diagram of SVM
from sklearn.linear_model import SGDClassifier
#title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
title = "Learning Curves (SVM, linear kernel)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
#estimator = SVC(gamma=0.001)
print("Thu the tham said: Creating diagram, SVM")
estimator = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter_no_change=5, random_state=42, verbose=0)
plot_learning_curve(estimator, title, X, y, ylim=(0.0, 1.0), cv=cv, n_jobs=8)
plt.show()

