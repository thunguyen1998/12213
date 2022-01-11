#Load in our librabries
from sklearn.datasets import fetch_20newsgroups
twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
twenty_test = fetch_20newsgroups(subset='test', shuffle=True)
print(type(twenty_train))
print("Thư thể thảm said: Đã khởi tạo mảng dữ liệu")
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

# Dưới đây là các phép đo lỗi phổ biến cho bài toán regression
# các phép đo error này nếu giá trị càng nhỏ tức là mô hình càng fit tốt với dữ liệu
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Các phép đo phổ biến cho bài toán classification
from sklearn.metrics import accuracy_score, f1_score

import warnings
warnings.filterwarnings('ignore')

####################
import pandas as pd

from tkinter import Tk
from tkinter import filedialog

Tk().withdraw()
openfilename = filedialog.askopenfilename(
title='slect_file_csv', initialdir='./')
data = pd.read_csv(openfilename)
#data = pd.read_csv("C:\Simulation\Code\HDP.csv")
x = data.iloc[:, 0]
y = data.iloc[:, 1]
print("Thư thể thảm said: Đã đọc file data")

    # Khởi tạo bộ train và bộ test, kiểm tra kích thước 
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state = 42)
print("Kích thước bộ x_train, y_train, x_test, y_test")
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    #gán dữ liệu vào code mẫu
twenty_train.data = x_train
twenty_test.data = x_test
twenty_train.target = y_train
twenty_test.target = y_test

print("Thư thể thảm said: kích thước bộ train và test")
print(len(twenty_train.data)) 
print(len(twenty_test.data))
# print(twenty_train.data[0])

# Câu 1: Tiền xử lý dữ liệu sử dụng Bag of Words và Tf-idf (sklearn)
# Extracting features from text files
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)

# TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print("Thư thể thảm said: Đã tạo bag of words có kích thước")
print(X_train_tfidf.shape)

# Câu 2: Sử dụng mô hình Naive Bayes cơ bản cho phân loại tin tức, đánh giá mô hình
# fit a Naive Bayes model to the data

from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB

#model = MultinomialNB()
#model.fit(X_train_tfidf, twenty_train.target)
#print(model)

# make predictions
#expected = twenty_train.target
#print(expected)

#predicted = model.predict(X_train_tfidf)
#print(predicted)

# summarize the fit of the model
#print(metrics.classification_report(expected, predicted))

# Câu hỏi 3 (nâng cao): Sử dụng Scikit-learn Pipelines để kết hợp quá trình (huấn luyện đồng thời) tiền xử lý dữ liệu và Naive Bayes

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import ShuffleSplit

## Huấn luyện mô hình
# Building a pipeline: We can write less code and do all of the above, by building a pipeline as follows:
# The names ‘vect’ , ‘tfidf’ and ‘clf’ are arbitrary but will be used later.
# We will be using the 'text_clf' going forward.
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
print("Thư thể thảm said: Bắt đầu train mô hình Naive bayes, không sử dụng stopword và stemming")
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)
print("Thư thể thảm said: Đã train xong mô hình Naive bayes, không sử dụng stopword và stemming")
print("Thư thể thảm said: Đang vẽ biểu đồ")
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
  
  """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
  
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
print("Thư thể thảm said: tắt biểu đồ để tiếp tục")
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
X, y = X_train_tfidf, twenty_train.target
plot_learning_curve(estimator, title, X, y, ylim=(0.0, 1.01), cv=cv, n_jobs=8)
plt.show()

# Câu 4 (nâng cao): Cải tiến mô hình ở câu 3 bằng cách loại bỏ những từ thông dụng và sử dụng GridSearchCV

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
print("Thư thể thảm said: Bắt đầu train mô hình Naive bayes, có sử dụng stopword và stemming")
text_mnb_stemmed = text_mnb_stemmed.fit(twenty_train.data, twenty_train.target)
print("Thư thể thảm said: Đã train xong mô hình Naive bayes, có sử dụng stopword và stemming")
predicted_mnb_stemmed = text_mnb_stemmed.predict(twenty_test.data)

np.mean(predicted_mnb_stemmed == twenty_test.target)




#Training Decision Tree and its performance
    # Import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
    # Import accuracy_score
from sklearn.metrics import accuracy_score
text_clf_dt = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-dt', DecisionTreeClassifier(random_state=42))])
print("Thư thể thảm said: Bắt đầu train mô hình Decision Tree, có sử dụng stopword và stemming")
text_clf_dt = text_clf_dt.fit(twenty_train.data, twenty_train.target)
print("Thư thể thảm said: Đã train mô hình Decision Tree, có sử dụng stopword và stemming")
predicted_dt = text_clf_dt.predict(twenty_test.data)
np.mean(predicted_dt == twenty_test.target)




#Training Random Forest and calculating its performance
    # Import RandomForestClassifier, VotingClassifier
from sklearn.ensemble import RandomForestClassifier
text_clf_rf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-dt', RandomForestClassifier(random_state=42))])
print("Thư thể thảm said: Bắt đầu train mô hình Randomforest, có sử dụng stopword và stemming")
text_clf_rf = text_clf_rf.fit(twenty_train.data, twenty_train.target)
print("Thư thể thảm said: Đã train mô hình Random Forest, có sử dụng stopword và stemming")
predicted_rf = text_clf_rf.predict(twenty_test.data)
np.mean(predicted_rf == twenty_test.target)





#Training KNN and calculating its performance
from sklearn.neighbors import KNeighborsClassifier
text_clf_KNN = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-svm', KNeighborsClassifier(n_neighbors=5))])
print("Thư thể thảm said: Bắt đầu train mô hình KNN, có sử dụng stopword và tremming")
text_clf_KNN = text_clf_KNN.fit(twenty_train.data, twenty_train.target)
print("Thư thể thảm said: Đã train mô hình SVM, có sử dụng stopword và tremming")
predicted_KNN = text_clf_KNN.predict(twenty_test.data)
np.mean(predicted_KNN == twenty_test.target)




# Training Support Vector Machines - SVM and calculating its performance
from sklearn.linear_model import SGDClassifier
text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter_no_change=5, 
                                                   random_state=42))])
print("Thư thể thảm said: Bắt đầu train mô hình SVM, có sử dụng stopword và tremming")
text_clf_svm = text_clf_svm.fit(twenty_train.data, twenty_train.target)
print("Thư thể thảm said: Đã train mô hình SVM, có sử dụng stopword và tremming")
predicted_svm = text_clf_svm.predict(twenty_test.data)
np.mean(predicted_svm == twenty_test.target)




#Creating Learning Curves Diagram of NB
estimator = MultinomialNB()
title = "Learning Curves (Naive Bayes)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
print("Thư thể thảm said: Đang vẽ biểu đồ, Naive Bayes")
X, y = X_train_tfidf, twenty_train.target
plot_learning_curve(estimator, title, X, y, ylim=(0.0, 1.0), cv=cv, n_jobs=8)


#Creating Learning Curves Diagram of Decision Tree
estimator = DecisionTreeClassifier()
title = "Learning Curves (Decision Tree)"
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
print("Thư thể thảm said: Đang vẽ biểu đồ, Decision Tree")
plot_learning_curve(estimator, title, X, y, ylim=(0.0, 1.0), cv=cv, n_jobs=8)

#Creating Learning Curves Diagram of RandomForest
estimator = RandomForestClassifier()
title = "Learning Curves (RandomForest)"
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
print("Thư thể thảm said: Đang vẽ biểu đồ, RandomForest")
plot_learning_curve(estimator, title, X, y, ylim=(0.0, 1.0), cv=cv, n_jobs=8)

#Creating Learning Curves Diagram of HNN
estimator = KNeighborsClassifier()
title = "Learning Curves (KNN)"
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
print("Thư thể thảm said: Đang vẽ biểu đồ, KNN")
plot_learning_curve(estimator, title, X, y, ylim=(0.0, 1.0), cv=cv, n_jobs=8)

#Creating Learning Curves Diagram of SVM
from sklearn.linear_model import SGDClassifier
#title = "Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
title = "Learning Curves (SVM, linear kernel)"
# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
#estimator = SVC(gamma=0.001)
print("Thư thể thảm said: Đang vẽ biểu đồ, SVM")
estimator = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter_no_change=5, random_state=42, verbose=0)
plot_learning_curve(estimator, title, X, y, ylim=(0.0, 1.0), cv=cv, n_jobs=8)
plt.show()


#Evaluating and confusion matrix
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix

#Confusion matrix of SVM
plot_confusion_matrix(text_clf_svm, twenty_test.data, twenty_test.target)  
plt.show()

#Confusion matrix of Decision Tree
plot_confusion_matrix(text_clf_dt, twenty_test.data, twenty_test.target)  
plt.show()

#Confusion matrix of RandomForest
plot_confusion_matrix(text_clf_rf, twenty_test.data, twenty_test.target)  
plt.show()

#Confusion matrix of Decision Tree
plot_confusion_matrix(text_clf_dt, twenty_test.data, twenty_test.target)  
plt.show()

#Confusion matrix of KNN
plot_confusion_matrix(text_clf_KNN, twenty_test.data, twenty_test.target)  
plt.show()


#Evaluation matrix of NB
print("Thư thể thảm said: Kết quả của NB + Stopwords")
# make predictions
expected = twenty_test.target
#print(expected)
predicted = text_mnb_stemmed.predict(twenty_test.data)
#print(predicted)
# summarize the fit of  model
print(metrics.classification_report(expected, predicted))


#Evaluation matrix of Decision Tree
print("Thư thể thảm said: Kết quả của Decision Tree + Stopwords")
# make predictions
expected = twenty_test.target
#rint(expected)
predicted = text_clf_dt.predict(twenty_test.data)
#print(predicted)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))


#Evaluation matrix of Random Forest
print("Thư thể thảm said: Kết quả của Random Forest + Stopwords")
# make predictions
expected = twenty_test.target
#rint(expected)
predicted = text_clf_rf.predict(twenty_test.data)
#print(predicted)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))


#Evaluation matrix of KNN
print("Thư thể thảm said: Kết quả của Random Forest + Stopwords")
# make predictions
expected = twenty_test.target
#rint(expected)
predicted = text_clf_KNN.predict(twenty_test.data)
#print(predicted)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))


#Evaluation matrix of SVM
print("Thư thể thảm said: Kết quả của SVM + Stopwords")
# make predictions
expected = twenty_test.target
#rint(expected)
predicted = text_clf_svm.predict(twenty_test.data)
#print(predicted)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))


