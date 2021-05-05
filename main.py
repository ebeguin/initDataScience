# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging as lg
from sklearn.model_selection import train_test_split


def plot_loyers_surface(matrice_loyers_surface, *args):
    plt.plot ( matrice_loyers_surface['surface'], matrice_loyers_surface['loyer'], 'ro', markersize=4 )
    if len(args)>=1:
        lg.debug("theta :   {}".format(args[0]))
        plt.plot ( [0, 250], [args[0].item ( 0 ), args[0].item ( 0 ) + 250 * args[0].item ( 1 )], linestyle='--',
                   c='#000000' )
    plt.show ()


def read_file (filename, lg):
    house_data = pd.read_csv ( filename )
    # affichons le nuage de points dont on dispose
    # filtering
    house_data_10000 = house_data[house_data['loyer'] < 10000]
    return house_data_10000

def vectorize_data (house_data, lg):
    x = np.matrix ( [np.ones ( house_data.shape[0] ), house_data['surface'].values] ).T
    lg.debug ( " x = {}".format ( x ) )
    y = np.matrix ( house_data['loyer'] ).T
    lg.debug ( " y = {}".format ( y ) )
    return x,y

def linear_basic(x, y, house_data, lg):
    theta_lb = np.linalg.inv ( x.T.dot ( x ) ).dot ( x.T ).dot ( y )
    lg.info ( "theta : {}".format ( theta_lb ) )
    return theta_lb


if __name__ == '__main__':
    # chargeons le dataset
    lg.basicConfig ( filename='ds.log', encoding='utf-8', level=lg.DEBUG )
    house_data = read_file ('house.csv',lg)
    x,y = vectorize_data(house_data,lg)
    theta_lb = linear_basic(x,y,house_data,lg)
    plot_loyers_surface ( house_data )
    plot_loyers_surface ( house_data, theta_lb )
    xtrain, xtest, ytrain, ytest = train_test_split(x,y, train_size=0.8)
