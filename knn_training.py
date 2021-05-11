from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn import neighbors
import numpy as np
import logging as lg
import matplotlib.pyplot as plt


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
prediction_errors = []
for k in range(2,15):
    model = neighbors.KNeighborsClassifier (n_neighbors=k)
    model.fit(xtrain, ytrain)
    prediction_errors.append(1 - model.score (xtest, ytest ))

#nombre de voisins le plus performant
neighbors_number_lowest_error = prediction_errors.index(min(prediction_errors)) +2

plt.plot(range(2,15), prediction_errors, 'o-')
plt.show()
print(neighbors_number_lowest_error)
#on récupère le classifieur le plus performant
model = neighbors.KNeighborsClassifier(n_neighbors=neighbors_number_lowest_error)
model.fit(xtrain,ytrain)

#on rédupère les predictions
predicted = model.predict(xtest)
#on redimmensionne sous forme d'images
images=xtest.reshape((-1, 28, 28))

#echantillon de 12 images
select = np.random.randint(images.shape[0], size=12)

#on affiche les images
fig,ax = plt.subplots(3,4)
for index, value in enumerate(select):
    plt.subplot(3,4,index+1)
    plt.axis('off')
    plt.imshow(images[value], interpolation="nearest")
    plt.title('Predicted : {}'.format(predicted[value]))

plt.show()






