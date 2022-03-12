from land_use_classification import open_data, load_cells, load_centroids, convert_to_residual, attach_to_centroids
from tweet_processing import residual_base
from open_cdr import merge_countries
from scipy.stats import zscore
from graph_cell import get_centroid_ind
import json
import project_utils as utils
import neuralnetwork as nn
import numpy as np
import pandas as pd
import datetime as dt
import math

def relu(input):
    return np.maximum(input,0)

def relu_prime_single(item):
    if item > 0 :
        return 1
    elif item < 0:
        return 0

def relu_prime(input):
    return np.array(list(map(relu_prime_single, input)))

def sigmoid(input):
    return 1 / (1 + np.exp(-input))

def sigmoid_prime(input):
    return sigmoid(input)*(1- sigmoid(input))

def load_nn_cells(datapoints:[(int,int,int)]):
    """ID, day, month"""
    centroids = load_centroids()
    cells = load_cells()
    cells, weekday, weekend = convert_to_residual(cells)
    centroids = attach_to_centroids(cells, centroids)
    out = []
    for datapoint in datapoints:
        print(datapoint)
        wanted_centroid = get_centroid_ind(centroids, datapoint[0])
        df = merge_countries("2013-%s-%s" % (str(datapoint[2]).zfill(2), str(datapoint[1]).zfill(2)))
        df = df[df[0]==datapoint[0]]
        df[8] = df[8] = (pd.to_datetime(df[1],unit='ms')+dt.timedelta(hours = 1)).dt.strftime("%w")
        df[9] = (pd.to_datetime(df[1],unit='ms')+dt.timedelta(hours = 1)).dt.strftime("%H")
        day = df[8].iloc[0]
        df = df.groupby([9]).sum().reset_index()
        sd = df[7].std()
        mean = df[7].mean()
        df[7] = (df[7]-mean)/sd
        temp_array = df[7].to_numpy()
        for i in range(24):
            if day == 0 or day == 6:
                temp_array[i] -= centroids[wanted_centroid].weekend[i]
            else:
                temp_array[i] -= centroids[wanted_centroid].weekday[i]
        out.append([temp_array])
    return np.array(out)

def load_nn_tweets(words:[str], dates: [int]):
    """dates in time since project epoch"""
    out = []
    distro = []
    base = residual_base()
    for i in words:
        distro.append([])
    for i in range(61):
        print(i)
        temp = []
        for j in range(len(words)):
            temp.append(0)
        if i < 30:
            with open(str(i + 1).zfill(2)+"1113tweets.json","r") as f:
                tweets = tweets = json.load(f)["tweets"]
                for tweet in tweets:
                    for j in range(len(words)):
                        temp[j] += tweet["text"].count(words[j])
        else:
            with open(str(i-29).zfill(2)+"1213tweets.json","r") as f:
                tweets = tweets = json.load(f)["tweets"]
                for tweet in tweets:
                    for j in range(len(words)):
                        temp[j] += tweet["text"].count(words[j])
        for j in range(len(words)):
            distro[j].append(temp[j])
    for i in range(len(words)):
        mean = sum(distro[i])/61
        temp_tot = 0
        for j in range(61):
            temp_tot = (distro[i][j]-mean)**2
        sd = np.sqrt(temp_tot/61)
        for j in range(61):
            distro[i][j] = (distro[i][j]-mean)/sd - base[j]
    for date in dates:
        temp_out = []
        for i in range(len(words)):
            temp_out.append(distro[i][date])
        out.append(temp_out)
    return np.array(out)


def cell_network():
    net = nn.network()
    net.add(nn.fclayer(24,16))
    net.add(nn.activation(sigmoid, sigmoid_prime))
    net.add(nn.fclayer(16,1))
    net.add(nn.activation(sigmoid, sigmoid_prime))
    expected = np.array([[1],[1],[1],[1],[0],[0],[0],[0],[0]])
    train_data = load_nn_cells([(5638,15,11),(5638,9,11),(5638,23,11),(5638,1,12),(5638,1,11),(5638,30,11),(5638,25,12),(5638,28,12),(5638,30,12)])
    net.loss_use(nn.mse,nn.mse_prime)
    net.train(train_data,expected,1000,0.1)
    test_data = load_nn_cells([(5638,8,12),(5638,22,12),(5638,2,11),(5638,16,12),(5638,11,12),(5638,26,12),(5638,29,11),(5638,15,12),(5638,28,12),(5638,3,11)])
    predicted = net.predict(test_data)
    actual = [1,1,1,1,1,0,0,0,0,0]
    for i in range(len(actual)):
        print("Actual : " + str(actual[i]) + " , Predicted : " + str(predicted[i][0]))
if __name__ == "__main__":
    #temp = np.arange(-10,10)
    #print(load_nn_tweets(["inter","milan"], [0,1,8]))
    cell_network()