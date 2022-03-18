from land_use_classification import open_data, load_cells, load_centroids, convert_to_residual, attach_to_centroids
from tweet_processing import residual_base
from open_cdr import merge_countries
from graph_cell import get_centroid_ind
import matplotlib.pyplot as plt
import pickle
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
        out.append([temp_out])
    print(out)
    return np.array(out)


def cell_network():
    net = nn.network()
    net.add(nn.fclayer(24,16))
    net.add(nn.activation(sigmoid, sigmoid_prime))
    net.add(nn.fclayer(16,1))
    net.add(nn.activation(sigmoid, sigmoid_prime))
    expected = [[1],[1],[1],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[1],[1]]
    train_data = [(5638,15,11),(5638,9,11),(5638,23,11),(5638,1,12),(5638,1,11),(5638,30,11),(5638,25,12),(5638,28,12),(5638,30,12),(6556,3,11),(6556,11,11),(6556,1,11),(6557,3,11),(6557,11,11),(6557,1,11),(6556,23,11),(6557,23,11),(6556,25,11),(6557,25,11)]
    for i in range(1,31):
        expected += [[0],[0],[0],[0]]
        train_data += [(3788,i,11),(3789,i,11),(3688,i,11),(3689,i,11)]
        expected += [[0],[0],[0],[0]]
        train_data += [(3788,i,12),(3789,i,12),(3688,i,12),(3689,i,12)]
    #train_data += [(2142,18,11),(2143,18,11),(2144,18,11),(2042,18,11),(2043,18,11),(2044,18,11)]
    #expected += [[1],[1],[1],[1],[1],[1]]
    expected = np.array(expected)
    train_data = load_nn_cells(train_data)
    net.loss_use(nn.mse,nn.mse_prime)
    net.train(train_data,expected,1000,0.1)
    test_data = load_nn_cells([(5638,8,12),(5638,22,12),(5638,2,11),(5638,16,12),(5638,11,12),(5638,26,12),(5638,29,11),(5638,15,12),(5638,28,12),(5638,3,11)])
    predicted = net.predict(test_data)
    actual = [1,1,1,1,1,0,0,0,0,0]
    for i in range(len(actual)):
        print("Actual : " + str(actual[i]) + " , Predicted : " + str(predicted[i][0]))
    return net

def tweet_network():
    words = load_nn_tweets(["inter","parma","sampdoria","livorno","ajax","roma","fiorentina","genoa"],[1,8,22,30,37,51,45,41,2,3,50,55,56,57])
    actuals = np.array([[[1]],[[1]],[[1]],[[1]],[[1]],[[1]],[[1]],[[1]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]]])
    net = nn.network()
    net.add(nn.fclayer(8,5))
    net.add(nn.activation(sigmoid,sigmoid_prime))
    net.add(nn.fclayer(5,1))
    net.add(nn.activation(sigmoid, sigmoid_prime))
    net.loss_use(nn.mse,nn.mse_prime)
    net.train(words,actuals,1000,0.1)
    test = load_nn_tweets(["inter","parma","sampdoria","livorno","ajax","roma","fiorentina","genoa"],[4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,53,54])
    test_actual = [[[0]]]*19
    predict = net.predict(test)
    for i in range(len(test_actual)):
        print("Actual : " + str(test_actual[i]) + " , Predicted : " + str(predict[i][0]))

def detect_events(network: nn.network, start_date : int):
    centroids = load_centroids()
    cells = load_cells()
    cells, weekday, weekend = convert_to_residual(cells)
    centroids = attach_to_centroids(cells, centroids)
    events = load_events_cell()
    centroid_id = {}
    for i in range(len(centroids)):
        for cell in centroids[i].cells:
            centroid_id[cell.id] = i
    for i in range(start_date, 61):
        if i < 30:
            df = merge_countries("2013-%s-%s" % ("11", str(i+1).zfill(2)))
            for j in range(1, 10001):
                temp_df = df[df[0]==j]
                if not temp_df.empty:
                    temp_df[8] = (pd.to_datetime(temp_df[1],unit='ms')+dt.timedelta(hours = 1)).dt.strftime("%w")
                    temp_df[9] = (pd.to_datetime(temp_df[1],unit='ms')+dt.timedelta(hours = 1)).dt.strftime("%H")
                    day = temp_df[8].iloc[0]
                    temp_df = temp_df.groupby([9]).sum().reset_index()
                    sd = temp_df[7].std()
                    mean = temp_df[7].mean()
                    temp_df[7] = (temp_df[7]-mean)/sd
                    temp_array = temp_df[7].to_numpy()
                    hours = temp_df[9].to_numpy()
                else:
                    hours = np.array([])
                    temp_array = hours = np.array([])
                for k in range(24):
                    if str(k).zfill(2) in hours:
                        if day == 0 or day == 6:
                            temp_array[k] -= centroids[centroid_id[j]].weekend[k]
                        else:
                            temp_array[k] -= centroids[centroid_id[j]].weekday[k]
                    elif k >= len(temp_array):
                        if day == 0 or day == 6:
                            temp_array = np.append(temp_array, [- centroids[centroid_id[j]].weekend[k]])
                        else:
                            temp_array = np.append(temp_array, [- centroids[centroid_id[j]].weekday[k]])
                    else:
                        if day == 0 or day == 6:
                             temp_array = np.insert(temp_array,k, [- centroids[centroid_id[j]].weekend[k]])
                        else:
                             temp_array = np.insert(temp_array,k, [- centroids[centroid_id[j]].weekday[k]])
                prediction = network.predict(np.array([[temp_array]]))
                if prediction[0][0] > 0.5:
                    events[j].append(i)
        else:
            df = merge_countries("2013-%s-%s" % ("12", str(i-29).zfill(2)))
            #don't forget to reset range
            for j in range(1, 10001):
                temp_df = df[df[0]==j]
                if not temp_df.empty:
                    temp_df[8] = (pd.to_datetime(temp_df[1],unit='ms')+dt.timedelta(hours = 1)).dt.strftime("%w")
                    temp_df[9] = (pd.to_datetime(temp_df[1],unit='ms')+dt.timedelta(hours = 1)).dt.strftime("%H")
                    day = temp_df[8].iloc[0]
                    temp_df = temp_df.groupby([9]).sum().reset_index()
                    sd = temp_df[7].std()
                    mean = temp_df[7].mean()
                    temp_df[7] = (temp_df[7]-mean)/sd
                    temp_array = temp_df[7].to_numpy()
                    hours = temp_df[9].to_numpy()
                else:
                    hours = np.array([])
                    temp_array = hours = np.array([])
                for k in range(24):
                    if str(k).zfill(2) in hours:
                        if day == 0 or day == 6:
                            temp_array[k] -= centroids[centroid_id[j]].weekend[k]
                        else:
                            temp_array[k] -= centroids[centroid_id[j]].weekday[k]
                    elif k >= len(temp_array):
                        if day == 0 or day == 6:
                            temp_array = np.append(temp_array, [- centroids[centroid_id[j]].weekend[k]])
                        else:
                            temp_array = np.append(temp_array, [- centroids[centroid_id[j]].weekday[k]])
                    else:
                        if day == 0 or day == 6:
                             temp_array = np.insert(temp_array,k, [- centroids[centroid_id[j]].weekend[k]])
                        else:
                             temp_array = np.insert(temp_array,k, [- centroids[centroid_id[j]].weekday[k]])
                prediction = network.predict(np.array([[temp_array]]))
                if prediction[0][0] > 0.5:
                    events[j].append(i)
        save_events_cell(events)
        print(i)
        
def graph_error():
    validation_data = load_nn_cells([(5638,8,12),(5638,22,12),(5638,2,11),(5638,16,12),(5638,11,12),(5638,26,12),(5638,29,11),(5638,15,12),(5638,28,12),(5638,3,11)])
    validation_actual = np.array([[1],[1],[1],[1],[1],[0],[0],[0],[0],[0]])
    expected = [[1],[1],[1],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[1],[1]]
    train_data = [(5638,15,11),(5638,9,11),(5638,23,11),(5638,1,12),(5638,1,11),(5638,30,11),(5638,25,12),(5638,28,12),(5638,30,12),(6556,3,11),(6556,11,11),(6556,1,11),(6557,3,11),(6557,11,11),(6557,1,11),(6556,23,11),(6557,23,11),(6556,25,11),(6557,25,11)]
    for i in range(1,31):
        expected += [[0],[0],[0],[0]]
        train_data += [(3788,i,11),(3789,i,11),(3688,i,11),(3689,i,11)]
        expected += [[0],[0],[0],[0]]
        train_data += [(3788,i,12),(3789,i,12),(3688,i,12),(3689,i,12)]
    #train_data += [(2142,18,11),(2143,18,11),(2144,18,11),(2042,18,11),(2043,18,11),(2044,18,11)]
    #expected += [[1],[1],[1],[1],[1],[1]]
    expected = np.array(expected)
    train_data = load_nn_cells(train_data)
    x = []
    y_train = []
    y_validation = []
    for i in range(1,21):
        net = nn.network()
        net.add(nn.fclayer(24,16))
        net.add(nn.activation(sigmoid, sigmoid_prime))
        net.add(nn.fclayer(16,1))
        net.add(nn.activation(sigmoid, sigmoid_prime))
        net.loss_use(nn.mse,nn.mse_prime)
        net.train(train_data,expected,i*10,0.1)
        train_predict = net.predict(train_data)
        validation_predicted = net.predict(validation_data)
        x.append(i*10)
        test_err = 0
        for j in range(len(expected)):
            test_err += net.loss(expected[j],train_predict[j])
        test_err /= len(y_train)
        validation_err = 0
        for j in range(len(validation_actual)):
            validation_err += net.loss(validation_actual[j],validation_predicted[j])
        validation_err /= len(validation_actual)
        y_train.append(test_err)
        y_validation.append(validation_err)
    fig, ax = plt.subplots()
    line_validation = ax.plot(x, y_validation)[0]
    line_train = ax.plot(x, y_train)[0]
    line_validation.set_label("Validation Error")
    line_train.set_label("Training Error")
    ax.legend()
    ax.set_ylabel("MSE Error")
    ax.set_xlabel("Number of Epochs")
    plt.show()
        

def save_events_cell(struct: dict):
    with open("cellevents.pkl","wb") as f:
        pickle.dump(struct, f)

def load_events_cell():
    with open("cellevents.pkl", "rb") as f:
        struct = pickle.load(f)
    return struct
        
def save_network_cell(struct: nn.network):
    with open("cellnetwork.pkl","wb") as f:
        pickle.dump(struct, f)

def load_network_cell():
    with open("cellnetwork.pkl", "rb") as f:
        struct = pickle.load(f)
    return struct

if __name__ == "__main__":
    #net = cell_network()
    #net = save_network_cell(net)
    #struct = {}
    #for i in range(1,10001):
    #    struct[i] = []
    #save_events_cell(struct)
    net = load_network_cell()
    detect_events(net, 0)
    #struct = load_events_cell()
    #print(struct)
    #graph_error()
