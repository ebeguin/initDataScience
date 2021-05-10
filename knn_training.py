from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn import neighbors
import numpy as np
import logging as lg

lg.basicConfig ( filename='ds.log', encoding='utf-8', level=lg.DEBUG )
# reading datset
mnist = fetch_openml('mnist_784', version=1)
#print images
lg.info("data : ".format(mnist.data))
#print annotations
lg.info("shape : ".format(mnist.target.shape))
sample = np.random.choice(70000, replace=True, size=5000)
sampled_data = mnist.data.iloc[sample]
sampled_target = mnist.target.iloc[sample]

# split between train and test
xtrain, xtest, ytrain, ytest = train_test_split(sampled_data, sampled_target, train_size=0.8)
model = neighbors.KNeighborsClassifier (n_neighbors=3)
model.fit(xtrain, ytrain)
lg.info ("xtest ::   {}".format(xtest))
first_index = xtest.first_valid_index()
lg.info (" prediction of {}".format(xtest.loc[first_index]))
prediction = model.predict([xtest.loc[first_index]])

lg.info ("prediction = {}".format(prediction))
print("prediction = {}".format(prediction))



