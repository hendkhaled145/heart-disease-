import pandas as pd
import numpy as np 
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.metrics import confusion_matrix

col_name =['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach','exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
dataset= pd.read_csv("C:/Users/20109/Desktop/4th_year/data_mining/section/DMproject/heart.csv")
print("Data:-")
print(dataset.head())
print()

#######split to features and label 
features_col=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach','exang', 'oldpeak', 'slope', 'ca', 'thal' ]
features=dataset[features_col]
label=dataset.target

######split to test and training 

features_train, features_test ,label_train ,label_test=train_test_split(features,label,test_size=0.3, random_state=1)
print("training data ")
print( features_train )
print("   ")
print(label_train)
print("   ")
print( "testing data")
print( features_test )
print("   ")
print(label_test)

#create a naive bayes classifier object 
nv = GaussianNB()

#fitting the data 
nv.fit(features_train, label_train )

#predict the respone for test dataset 
label_pred= nv.predict(features_test)
print("prediction data")
print(label_pred)

#model accuracy 
print("accuracy: ",metrics.accuracy_score(label_test, label_pred))
print("confusion matrix: " ,confusion_matrix(label_test, label_pred))
print(metrics.classification_report(label_test, label_pred))

pred=nv.predict(np.array([[52,1,0,125,212,0,1,168,0,1,2,2,3]]))
print("****",pred)