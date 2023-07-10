import os
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pickle

input_dir = r'C:\Users\mkmt724\Documents\Carparking\clf-data'
categories = ['empty','not_empty']

data = []
labels = []

for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir,category)):
        img_path = os.path.join(input_dir,category,file)
        #print(img_path)
        img = imread(img_path)
        img = resize(img,(15,15))
        data.append(img.flatten())
        labels.append(category_idx)
data = np.asarray(data)
labels = np.asarray(labels)
x_train,x_test,y_train,y_test = train_test_split(data,labels,test_size=.2,stratify=labels,shuffle=True)
# SVC
classifier = SVC()
# params
parameters = [{'gamma':[0.01,0.001,0.0001],'C':[10,100,1000]}]
# gridsearch_cv
gridsearch = GridSearchCV(classifier,parameters)
gridsearch.fit(x_train,y_train)
# best_estimaters
best_estimaters = gridsearch.best_estimator_
print('****best_estimaters:',best_estimaters)
y_prediction = best_estimaters.predict(x_test)
score  = accuracy_score(y_test,y_prediction)
print('{}% of samples where correctly classified'.format(str(score * 100)))
# pickling 
pickle.dump(best_estimaters,open('./model.p','wb'))