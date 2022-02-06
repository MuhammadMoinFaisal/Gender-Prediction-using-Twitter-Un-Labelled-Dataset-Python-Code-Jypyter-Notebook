
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
import pickle

import nltk

import re 

from nltk.stem import PorterStemmer # for stemming

from nltk.stem import WordNetLemmatizer # for lemmatization

from nltk.corpus import stopwords
nltk.download('punkt')

nltk.download('stopwords')
nltk.download('wordnet')
from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# # LOAD THE DATASET



data = pd.read_csv(r"C:\UpworkProjects\IOTA\tweets.csv")




data


# # PRE-PROCESSING THE DATASET



data.shape




data.info()




data.isnull().sum()


# # Choosing only the required columns



df = data[['tweet']]




df


# # Checking the Null Values



df.isnull().sum()


# # Removing the NULL VALUES



df.dropna(axis = 0, inplace = True)




df.isnull().sum()




df.shape




df['tweet'].value_counts()




df.dtypes


# # Cleaning the Description Column removing punctuation marks 



def clean(review):

    descrip = re.sub('[^a-zA-Z]', ' ', review)

    review = review.lower()

    return review

df['tweet'] = pd.DataFrame(df['tweet'].apply(lambda x: clean(x)))

df.head()

#Cleaning the data which includes punctuation removal,number removal, and different signs like ‘@’,'()’,’#’, and URL with ‘ ‘.


df['tweet'].replace('[@+]', "", regex=True,inplace=True)

df['tweet'].replace('[()]', "", regex=True,inplace=True)


df['tweet'].replace('[#+]', "", regex=True, inplace = True)


#url_regex = "(https?://)(s)*(www.)?(s)*((w|s)+.)*([w-s]+/)*([w-]+)((?)?[ws]*=s*[w%&]*)*"
#df['descrip_Cleaned'] = df['descrip_Cleaned'].replace(url_regex, "", regex=True)




#url_regex = "(https?://)(s)*(www.)?(s)*((w|s)+.)*([w-s]+/)*([w-]+)((?)?[ws]*=s*[w%&]*)*"
#df['descrip_Cleaned'] = df['descrip_Cleaned'].replace(url_regex, "", regex=True)




df['tweet'] = df['tweet'].replace(to_replace=r'^https?:\/\/.*[\r\n]*',value='',regex=True)




df['tweet'] = df['tweet'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])




print(df)




url_regex = "(https?://)" 
df['tweet'] = df['tweet'].replace(url_regex, "", regex=True)




print(df)


# # REMOVING STOP WORDS



from nltk.corpus import stopwords
stop = stopwords.words('english')




df['tweet'] = df['tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))




df




df.isnull().sum()




df.dropna(axis = 0, inplace = True)




df.isnull().sum()




df['tweet'] = df['tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))




df[['tweet']]


# # Vectorization of texts in ‘descrip_Cleaned’ and ‘text_Cleaned’ columns.



x= df['tweet']




#y = df['gender']
x




cv = CountVectorizer(max_features = 2000)
x = cv.fit_transform(df['tweet']).toarray()
#x1=cv.fit_transform(df['text_Cleaned']).toarray()




x.shape




filenameDT = r'C:\UpworkProjects\IOTA\picklefile\model_pt_f.sav'




# load the model from disk
model = pickle.load(open(filenameDT, 'rb'))




y_pred = model.predict(x)




y_pred




y_pred.dtype




y_pred.astype("object")




y_pred = y_pred.astype(str)




y_pred




y_pred[y_pred =="1.0"] = "male"
y_pred[y_pred== "0.0"] = "female"




import numpy
import sys
numpy.set_printoptions(threshold=sys.maxsize)


y_pred


y_pred = pd.DataFrame(y_pred, columns = ['gender'])




print(y_pred)




print(data)

print(df)

print(data)




data_labelled = pd.concat([data,y_pred],join='inner',axis=1)




data_labelled




df = data_labelled[['tweet', 'gender']]


# # Checking the Null Values



df.isnull().sum()


# # Removing the NULL VALUES



df.dropna(axis = 0, inplace = True)




df.isnull().sum()




df['gender'].value_counts()


# # Label Encoding, Representing Male with 1 and Female with 0



for gen in df['gender']:

  if gen=='male':

     df['gender'].replace({'male':1},inplace=True)

  elif gen=='female':

     df['gender'].replace({'female':0},inplace=True)

df['gender'].value_counts()


# # Cleaning the Description Column removing punctuation marks



def clean(review):

    descrip = re.sub('[^a-zA-Z]', ' ', review)

    review = review.lower()

    return review

df['tweet'] = pd.DataFrame(df['tweet'].apply(lambda x: clean(x)))

df.head()


# # Cleaning the data which includes punctuation removal,number removal, and different signs like ‘@’,'()’,’#’, and URL with ‘ ‘.



df['tweet'].replace('[@+]', "", regex=True,inplace=True)

df['tweet'].replace('[()]', "", regex=True,inplace=True)


df['tweet'].replace('[#+]', "", regex=True, inplace = True)




df['tweet'] = df['tweet'].replace(to_replace=r'^https?:\/\/.*[\r\n]*',value='',regex=True)




df['tweet'] = df['tweet'].apply(lambda x: re.split('https:\/\/.*', str(x))[0])


# # REMOVING STOP WORDS



from nltk.corpus import stopwords
stop = stopwords.words('english')




df['tweet'] = df['tweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))




cv = CountVectorizer(max_features = 2000)
x = cv.fit_transform(df['tweet']).toarray()




x.shape




y = df['gender']


# # Data was split into train and test data with an 80:20 ratio.



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state = 0)





gnbmodel = GaussianNB()
gnbmodel.fit(X_train , y_train)
y_pred = gnbmodel.predict(X_test)




accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))




confusion_matrix(y_test, y_pred)




print(classification_report(y_test, y_pred))


# # GridsearchCV() was used to tune hyperparameters for threse Classifiers.



param_grid_nb = {
    'var_smoothing': np.logspace(0,-9, num=100)
}



#nbModel_grid = GridSearchCV(estimator=gnbmodel, param_grid=param_grid_nb, verbose=1, cv=3, n_jobs=-1)



#bModel_grid.fit(X_train, y_train)




#nbModel_grid.best_params_


#y_pred_hyper = nbModel_grid.predict(X_test)




#print(confusion_matrix(y_test, y_pred_hyper), ": is the confusion matrix")




#accuracy = accuracy_score(y_test, y_pred_hyper)
#print("Accuracy: %.2f%%" % (accuracy * 100.0))




#print(classification_report(y_test, y_pred_hyper))


# # Lightgbm Classifier



lgbmodel = LGBMClassifier(max_depth=3)
lgbmodel.fit(X_train, y_train)




y_pred1= lgbmodel.predict(X_test)




accuracy = accuracy_score(y_test, y_pred1)

print("Accuracy: %.2f%%" % (accuracy * 100.0))


# # Hyper parameter tuning using Grid Search CV



param_grid = {
              "max_depth": [2, 3, 5, 10],
              "min_child_weight": [0.001, 0.002],
              "learning_rate": [0.05, 0.1]
              }




lgbgrid = GridSearchCV(estimator = lgbmodel, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 0)

lgbgrid.fit(X_train, y_train)




lgbgrid.best_params_




y_pred1_hyper = lgbgrid.predict(X_test)




accuracy = accuracy_score(y_test, y_pred1_hyper)
print("Accuracy: %.2f%%" % (accuracy * 100.0))




print(classification_report(y_test, y_pred1_hyper))


# # XGBoost Classifier



xgbmodel = XGBClassifier(max_depth=5, min_child_weight=1)
xgbmodel.fit(X_train, y_train)




y_pred2 = xgbmodel.predict(X_test)




accuracy = accuracy_score(y_test, y_pred2)
print("Accuracy: %.2f%%" % (accuracy * 100.0))




print(classification_report(y_test, y_pred2))


# # Hyperparameter tuning of XGBoost Classifier using GridsearchCV() function.



xgb_param_grid = {
              "max_depth": [3, 5],
              "min_child_weight": [1, 2],
              }




xgbgrid = GridSearchCV(estimator = xgbmodel, param_grid = xgb_param_grid, cv = 3, n_jobs = -1, verbose = 0)

xgbgrid.fit(X_train, y_train)




xgbgrid.best_params_




y_pred2_hyper = xgbgrid.predict(X_test)




y_pred2_hyper




accuracy = accuracy_score(y_test, y_pred2_hyper)
print("Accuracy: %.2f%%" % (accuracy * 100.0))




print(classification_report(y_test, y_pred2_hyper))

