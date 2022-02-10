import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

col_name =['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach','exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
dataset= pd.read_csv("C:/Users/20109/Desktop/4th_year/data_mining/section/DMproject/heart.csv")
print("Data:-")
print(dataset.head())
print()

#######split to features and label 

features=dataset.iloc[:,:-1]
label=dataset.iloc[:,13]

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


#clf=DecisionTreeClassifier()
clf=DecisionTreeClassifier(criterion="entropy",max_depth=3)

clf=clf.fit(features_train,label_train)

label_pred=clf.predict(features_test)
print("prediction data")
print(label_pred)

#model accuracy 
print("accuracy: ",metrics.accuracy_score(label_test, label_pred))
print("confusion matrix: " ,confusion_matrix(label_test, label_pred))
print(metrics.classification_report(label_test, label_pred))


pred=clf.predict(np.array([[52,1,0,125,212,0,1,168,0,1,2,2,3]]))
print("****",pred)


plt.figure(figsize=(25,10))
a=plot_tree(clf, feature_names=col_name.remove("target"),class_names=['0','1'],filled=True,rounded=True,fontsize=14)

