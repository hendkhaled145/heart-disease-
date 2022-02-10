import pandas as ps
import numpy as np
from matplotlib import pyplot as plt

data= ps.read_csv("C:/Users/20109/Desktop/4th_year/data_mining/section/DMproject/heart.csv" , na_values="?")
#print(data.head())
#print(data.dtypes)
#print(data.describe())
#print(data["age"].value_counts())
print(data.columns)

#non_graphical
print(ps.crosstab(data["age"],data["target"]))

#histogram
plt.hist(data['age'])
plt.xlabel("age")
plt.ylabel("Count")
plt.title("histogram of Age")
plt.show()

#scatter plot
plt.scatter(data['age'],data['oldpeak'])
plt.xlabel("age")
plt.ylabel("Number of oldpeak")
plt.title("Number of oldpeak with age")
plt.show()