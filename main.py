# Reading data
import pandas as pd
import numpy as np

path = r'reviews.csv'
df= pd.read_csv(path)
print(df.head())
print(df.shape)

# Data cleaning and pre porocessing

import re
import nltk
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
allStopwords=stopwords.words('english')
allStopwords.remove('not')

corpus=[]

for i in range(0, 9953):
  review = re.sub('[^a-zA-Z]', ' ', df['Review'][i])
  review = review.lower()
  review = review.split()
  review = [ps.stem(word) for word in review if not word in set(allStopwords)]
  review = ' '.join(review)
  corpus.append(review)





# Data transformation

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 700)

X = cv.fit_transform(corpus).toarray()
y = df.iloc[:, -1].values
print(len(X))
print(len(y))


# Saving BoW dictionary to later use in prediction

import pickle
bow_path = r'analysis.pkl'
pickle.dump(cv, open(bow_path, "wb"))

# Dividing df into training and test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


# Model fitting (Naive Bayes)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)



# Exporting NB Classifier to later use in prediction

import joblib
joblib.dump(classifier, r'classifier model')

# Model performance

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)

print(accuracy_score(y_test, y_pred))