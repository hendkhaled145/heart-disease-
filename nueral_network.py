import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

dataset= pd.read_csv("C:/Users/20109/Desktop/4th_year/data_mining/section/DMproject/heart.csv")

print("Data:-")
print(dataset.head())
print()

#######split to features and label 

features=dataset.iloc[:,:-1]
label=dataset.iloc[:,13]

###scale values all between 1 and 0(normalization)
min_max_scaler = preprocessing.MinMaxScaler()
features_scale = min_max_scaler.fit_transform(features)
print(features_scale)

######split to test and training 

features_train, features_test ,label_train ,label_test=train_test_split(features,label,test_size=0.3, random_state=1)

####split test data valset, testset
f_val, f_test, l_val, l_test = train_test_split(features_test, label_test, test_size=0.5)

######Setting up the Architecture
print(features_train.shape, f_val.shape, f_test.shape, label_train.shape, l_val.shape, l_test.shape)
model = Sequential([
    Dense(32, activation='relu', input_shape=(13,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid'),])
###Filling in the best numbers
model.compile(optimizer='sgd',loss='mean_squared_error',metrics=['accuracy'])

hist = model.fit(features_train, label_train,batch_size=32, epochs=100,validation_data=(f_val, l_val))

print(model.evaluate(f_test, l_test)[1])

#Visualizing Loss and Accuracy
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'test'], loc='upper right')
plt.show()

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'test'], loc='lower right')
plt.show()










