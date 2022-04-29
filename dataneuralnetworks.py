import fiona
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
import shapely
import geopandas as gpd

def project_time(day, month):
    """Converts from an integer day and month to project day
    
    Args
    day - day to convert
    month - motnh to convert
    
    Returns
    Integer representing time from project epoch"""
    if int(month) == 11:
        return int(day)-1
    else:
        return int(day)+29

def relu(input):
    """Relu activation function implementation
    
    Args
    input - value to find the relu value for
    
    Returns
    float"""
    return np.maximum(input,0)

def relu_prime_single(item):
    """Relu derivative implementation
    
    Args
    input - value to find the derivative of relu for
    
    Returns
    float"""
    if item > 0 :
        return 1
    elif item < 0:
        return 0

def relu_prime(input):
    """Applies relu derivative to a list
    
    Args
    input - list to find relu derivative for
    
    Returns
    list with relu derivative values"""
    return np.array(list(map(relu_prime_single, input)))

def sigmoid(input):
    """Sigmoid activation function implementation
    
    Args
    input - value to find the sigmoid value for
    
    Returns
    float"""
    return 1 / (1 + np.exp(-input))

def sigmoid_prime(input):
    """Sigmoid derivative implementation
    
    Args
    input - value to find the sigmoid derivative for
    
    Returns
    float"""
    return sigmoid(input)*(1- sigmoid(input))

def generate_test(event_list:[[int]]):
    out = []
    for events in event_list:
        this_list = []
        for i in range(61):
            if i in events:
                this_list.append(1)
            else:
                this_list.append(0)
        out.append([this_list])
    return np.array(out)

def load_nn_cells(datapoints:[(int,int,int)]):
    """Loads the cell data for the neural network
    
    Args
    datapoints - cells and dates to get data for in the format: ID, day, month
    
    Returns
    List of values in valid format for the neural network"""
    dates = set()
    date_cell = {}
    for datapoint in datapoints:
        dates.add((datapoint[1],datapoint[2]))
        if (datapoint[1],datapoint[2]) in date_cell.keys():
            date_cell[(datapoint[1],datapoint[2])].append(datapoint[0])
        else:
            date_cell[(datapoint[1],datapoint[2])] = [datapoint[0]]
    cdf = pd.read_csv("CensusDataZScore.csv", index_col = 0)
    centroids = load_centroids()
    cells = load_cells()
    cells, weekday, weekend = convert_to_residual(cells)
    centroids = attach_to_centroids(cells, centroids)
    values_dict = {}
    for date in dates:
        df = merge_countries("2013-%s-%s" % (str(date[1]).zfill(2), str(date[0]).zfill(2)))
        for cell in date_cell[date]:
            print([cell, date[0],date[1]])
            wanted_centroid = get_centroid_ind(centroids, datapoint[0])
            temp_df = df[df[0]==datapoint[0]]
            temp_df[8] = temp_df[8] = (pd.to_datetime(temp_df[1],unit='ms')+dt.timedelta(hours = 1)).dt.strftime("%w")
            temp_df[9] = (pd.to_datetime(temp_df[1],unit='ms')+dt.timedelta(hours = 1)).dt.strftime("%H")
            day = temp_df[8].iloc[0]
            temp_df = temp_df.groupby([9]).sum().reset_index()
            sd = temp_df[7].std()
            mean = temp_df[7].mean()
            temp_df[7] = (temp_df[7]-mean)/sd
            temp_array = temp_df[7].to_numpy()
            for i in range(24):
                if day == 0 or day == 6:
                    temp_array[i] -= centroids[wanted_centroid].weekend[i] -weekend[i]
                else:
                    temp_array[i] -= centroids[wanted_centroid].weekday[i] -weekday[i]
            temp_cdf = cdf[cdf["CellID"] == cell]
            temp_array = np.concatenate([temp_array,np.array([temp_cdf.iloc[0]["P1"],temp_cdf.iloc[0]["E3"],temp_cdf.iloc[0]["E4"]])])
            values_dict[(cell, date[0], date[1])] = [temp_array]
    out = []
    for datapoint in datapoints:
        out.append(values_dict[datapoint])
    return np.array(out)

def load_nn_cells_single(cell_list: [int]):
    """Created to load cells for an alternative format of the neural networks (daily resolution and select days)"""
    dates = set()
    date_cell = {}
    df = pd.read_csv("loose_merge.csv", index_col = 0)
    out = []
    cdf = pd.read_csv("CensusDataZScore.csv", index_col = 0)
    centroids = load_centroids()
    cells = load_cells()
    cells, weekday, weekend = convert_to_residual(cells)
    centroids = attach_to_centroids(cells, centroids)
    with open("centroid_pattern.json","r") as f:
        centroids_pattern = json.load(f)
    for cell in cell_list:
        print(cell)
        wanted_centroid = get_centroid_ind(centroids, cell-1)
        temp_df = df[df["0"] == cell]
        temp_df["8"] = (pd.to_datetime(temp_df["1"],unit='ms')+dt.timedelta(hours = 1)).dt.strftime("%d")
        temp_df["9"] = (pd.to_datetime(temp_df["1"],unit='ms')+dt.timedelta(hours = 1)).dt.strftime("%m")
        temp_df = temp_df.groupby(["9","8"]).sum().reset_index()
        temp_df["10"] = temp_df.apply(lambda x: project_time(x["8"],x["9"]),axis = 1)
        value_list = temp_df["7"].tolist()
        days_list = temp_df["10"].tolist()
        for i in range(61):
            if not i in days_list:
                if i == len(days_list):
                    value_list.append(0)
                else:
                    value_list.insert(i,0)
        value_list = np.array(value_list)
        std = value_list.std()
        mean = value_list.mean()
        value_list = (value_list-mean)/std
        value_list = value_list - np.array(centroids_pattern["centroids"][wanted_centroid]["pattern"])
        temp_cdf = cdf[cdf["CellID"] == cell]
        value_list = np.concatenate([value_list,np.array([temp_cdf.iloc[0]["P1"],temp_cdf.iloc[0]["E3"],temp_cdf.iloc[0]["E4"]])])
        out.append([value_list])
    return np.array(out)

    cdf = pd.read_csv("CensusDataZScore.csv", index_col = 0)
    centroids = load_centroids()
    cells = load_cells()
    cells, weekday, weekend = convert_to_residual(cells)
    centroids = attach_to_centroids(cells, centroids)
    values_dict = {}
    

def load_nn_tweets_single(words:[str]):
    """Created to load twitter data for an alternative format of the neural networks (daily resolution and select days of events)"""
    distro = []
    base = residual_base()
    for i in words:
        distro.append([])
    #Get word frequency
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
                        temp[j] += tweet["text"].lower().count(words[j])
        else:
            with open(str(i-29).zfill(2)+"1213tweets.json","r") as f:
                tweets = tweets = json.load(f)["tweets"]
                for tweet in tweets:
                    for j in range(len(words)):
                        temp[j] += tweet["text"].lower().count(words[j])
        for j in range(len(words)):
            distro[j].append(temp[j])
    print(distro)
    #Normalise
    for i in range(len(words)):
        mean = sum(distro[i])/61
        temp_tot = 0
        for j in range(61):
            temp_tot = (distro[i][j]-mean)**2
        sd = np.sqrt(temp_tot/61)
        for j in range(61):
            distro[i][j] = (distro[i][j]-mean)/sd - base[j]
    out = []
    for i in distro:
        out.append([i])
    return np.array(out)
  

def load_nn_tweets(words:[str], dates: [int]):
    """Loads the twitter data for the neural network
    
    Args
    words - words to get data for
    dates - dates to get data for in days since epoch
    
    Returns
    List of values in valid format for the neural network"""
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
                        temp[j] += tweet["text"].lower().count(words[j])
        else:
            with open(str(i-29).zfill(2)+"1213tweets.json","r") as f:
                tweets = tweets = json.load(f)["tweets"]
                for tweet in tweets:
                    for j in range(len(words)):
                        temp[j] += tweet["text"].lower().count(words[j])
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
    #print(out)
    return np.array(out)

def load_nn_combined(datapoints:[(int, int, int)], words: [str]):
    """Function to load data for a single neural network taking both twitter and mobile phone data"""
    dates = set()
    date_cell = {}
    for datapoint in datapoints:
        dates.add((datapoint[1],datapoint[2]))
        if (datapoint[1],datapoint[2]) in date_cell.keys():
            date_cell[(datapoint[1],datapoint[2])].append(datapoint[0])
        else:
            date_cell[(datapoint[1],datapoint[2])] = [datapoint[0]]
    cdf = pd.read_csv("CensusDataZScore.csv", index_col = 0)
    centroids = load_centroids()
    cells = load_cells()
    cells, weekday, weekend = convert_to_residual(cells)
    centroids = attach_to_centroids(cells, centroids)
    values_dict = {}
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
                        temp[j] += tweet["text"].lower().count(words[j])
        else:
            with open(str(i-29).zfill(2)+"1213tweets.json","r") as f:
                tweets = tweets = json.load(f)["tweets"]
                for tweet in tweets:
                    for j in range(len(words)):
                        temp[j] += tweet["text"].lower().count(words[j])
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
        df = merge_countries("2013-%s-%s" % (str(date[1]).zfill(2), str(date[0]).zfill(2)))
        epoch_date = 0
        if date[1] == 12:
            epoch_Date = date[0] + 29
        else:
            epoch_date = date[0] - 1
        word_distro = []
        for i in range(len(words)):
            word_distro.append(distro[i][epoch_date])
        for cell in date_cell[date]:
            print([cell, date[0],date[1]])
            wanted_centroid = get_centroid_ind(centroids, datapoint[0])
            temp_df = df[df[0]==datapoint[0]]
            temp_df[8] = temp_df[8] = (pd.to_datetime(temp_df[1],unit='ms')+dt.timedelta(hours = 1)).dt.strftime("%w")
            temp_df[9] = (pd.to_datetime(temp_df[1],unit='ms')+dt.timedelta(hours = 1)).dt.strftime("%H")
            day = temp_df[8].iloc[0]
            temp_df = temp_df.groupby([9]).sum().reset_index()
            sd = temp_df[7].std()
            mean = temp_df[7].mean()
            temp_df[7] = (temp_df[7]-mean)/sd
            temp_array = temp_df[7].to_numpy()
            for i in range(24):
                if day == 0 or day == 6:
                    temp_array[i] -= centroids[wanted_centroid].weekend[i] - weekend[i]
                else:
                    temp_array[i] -= centroids[wanted_centroid].weekday[i] - weekday[i]
            temp_cdf = cdf[cdf["CellID"] == cell]
            temp_array = np.concatenate([temp_array,np.array([temp_cdf.iloc[0]["P1"],temp_cdf.iloc[0]["E3"],temp_cdf.iloc[0]["E4"]]), np.array(word_distro)])
            values_dict[(cell, date[0], date[1])] = [temp_array]
    out = []
    for datapoint in datapoints:
        out.append(values_dict[datapoint])
    return np.array(out)
   

def cell_network():
    """Configures and trains the mobile phone neural network
    
    Returns
    Neural Network object"""
    net = nn.network()
    net.add(nn.fclayer(27,18))
    net.add(nn.activation(sigmoid, sigmoid_prime))
    net.add(nn.fclayer(18,3))
    net.add(nn.activation(sigmoid, sigmoid_prime))
    net.add(nn.fclayer(3,1))
    net.add(nn.activation(sigmoid, sigmoid_prime))
    net.loss_use(nn.mse,nn.mse_prime)
    expected = [[1],[1],[1],[1],[0],[0],[0],[0],[0]]
    train_data = [(5638,15,11),(5638,9,11),(5638,23,11),(5638,1,12),(5638,1,11),(5638,30,11),(5638,25,12),(5638,28,12),(5638,30,12)]
    train_data += [(3912, 31, 12),(3710, 31, 12), (1782, 5, 11), (2359, 2, 11), (2359, 16, 11), (2359, 30, 11), (5259, 1, 11), (5259,2,11), (5259,9,11), (7468,2,11), (4511, 5 ,11)]
    expected += [[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]
    expected = np.array(expected)
    train_data = load_nn_cells(train_data)
    net.train(train_data,expected,1000,0.1)
    test_data = load_nn_cells([(5638,8,12),(5638,22,12),(5638,2,11),(5638,16,12),(5638,11,12),(5638,26,12),(5638,29,11),(5638,15,12),(5638,28,12),(5638,3,11)])
    predicted = net.predict(test_data)
    actual = [1,1,1,1,1,0,0,0,0,0]
    for i in range(len(actual)):
        print("Actual : " + str(actual[i]) + " , Predicted : " + str(predicted[i][0]))
    return net

def cell_network_single():
    """Loads and trains an alternative approach to the neural network, where the network is given a set of data for the whole time frame and selects days on which events occur"""
    net = nn.network()
    net.add(nn.fclayer(64,61))
    net.add(nn.activation(sigmoid, sigmoid_prime))
    net.add(nn.fclayer(61,61))
    net.add(nn.activation(sigmoid, sigmoid_prime))
    net.loss_use(nn.mse,nn.mse_prime)
    train_data = load_nn_cells_single([5638,2359,2362,1982])
    train_actuals = generate_test([[8,14,22,1,30,37,51,45,40],[],[],[]])
    net.train(train_data,train_actuals,1000,0.1)
    test_data = load_nn_cells_single([5837,1615])
    predicted = net.predict(test_data)
    return net

def cell_network_san_siro_only():
    """Similar to cell_network but only a particular subset of data"""
    net = nn.network()
    net.add(nn.fclayer(27,18))
    net.add(nn.activation(sigmoid, sigmoid_prime))
    net.add(nn.fclayer(18,3))
    net.add(nn.activation(sigmoid, sigmoid_prime))
    net.add(nn.fclayer(3,1))
    net.add(nn.activation(sigmoid, sigmoid_prime))
    net.loss_use(nn.mse,nn.mse_prime)
    event_days = [(5638,15,11),(5638,9,11),(5638,1,12),(5638,8,12),(5638,22,12),(5638,23,11),(5638,2,11),(5638,16,12),(5638,11,12)]
    expected = [[1],[1],[1],[1],[1],[1],[1],[1],[1]]
    train_data = event_days
    test_data = []
    actual = []
    for i in range(1,31):
        test_data.append((5738,i,11))
        if not (5638, i, 11) in event_days:
            train_data += [(5638, i, 11)]
            expected += [[0]]
            actual += [[0]]
        else:
            actual += [[1]]
    for i in range(1,32):
        test_data.append((5738,i,12))
        if not (5638, i, 12) in event_days:
            train_data += [(5638, i, 12)]
            expected += [[0]]
            actual += [[0]]
        else:
            actual += [[1]]
    expected = np.array(expected)
    train_data = load_nn_cells(train_data)
    net.train(train_data,expected,10000,0.1)
    test_data = load_nn_cells(test_data)
    predicted = net.predict(test_data)
    for i in range(len(actual)):
        print("Actual : " + str(actual[i]) + " , Predicted : " + str(predicted[i][0]))
    return net

def tweet_network(word_list):
    """Configures the Twitter neural network and trains it
    
    Args
    word_list - list of words to train the neural network on
    
    Returns
    Neural network object"""
    words = load_nn_tweets(word_list,[8,22,30,37,51,41,2,3,50,55,56,57,5,6,32,12,45])
    actuals = np.array([[[1]],[[1]],[[1]],[[1]],[[1]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[0]],[[1]]])
    net = nn.network()
    net.add(nn.fclayer(len(word_list),int(2*len(word_list)/3)))
    net.add(nn.activation(sigmoid,sigmoid_prime))
    net.add(nn.fclayer(int(2*len(word_list)/3),3))
    net.add(nn.activation(sigmoid, sigmoid_prime))
    net.add(nn.fclayer(3,1))
    net.add(nn.activation(sigmoid, sigmoid_prime))
    net.loss_use(nn.mse,nn.mse_prime)
    net.train(words,actuals,1000,0.1)
    test_in = []
    for i in range(61):
        test_in.append(i)
    test = load_nn_tweets(word_list,test_in)
    predict = net.predict(test)
    for i in range(len(test)):
        print(str(i) + " : " + str(predict[i][0][0]))
    return net

def tweet_network_single_word():
    """Loads and trains an alternative approach to the neural network, where the network is given a set of data for the whole time frame and selects days on which events occur"""
    net = nn.network()
    net.add(nn.fclayer(61,61))
    net.add(nn.activation(sigmoid,sigmoid_prime))
    net.add(nn.fclayer(61,61))
    net.add(nn.activation(sigmoid, sigmoid_prime))
    net.loss_use(nn.mse,nn.mse_prime)
    input = load_nn_tweets_single(["inter", "parma", "arctic monkeys", "bastille"])
    expected_output = []
    temp = []
    for i in range(61):
        if i in [8,30,37,51]:
            temp.append(1)
        else:
            temp.append(0)
    expected_output.append(temp)
    temp = []
    for i in range(61):
        if i in [37]:
            temp.append(1)
        else:
            temp.append(0)
    expected_output.append(temp)
    temp = []
    for i in range(61):
        if i in [12]:
            temp.append(1)
        else:
            temp.append(0)
    expected_output.append(temp)
    temp = []
    for i in range(61):
        if i in [22]:
            temp.append(1)
        else:
            temp.append(0)
    expected_output.append(temp)
    print(expected_output)
    net.train(input,np.array(expected_output),10000,0.1)
    test = load_nn_tweets_single(["sampdoria","livorno", "pixies", "skrillex", "giuseppe meazza","bob dylan"])
    print(test)
    events=[]
    for i in test:
        events.append([])
    for i in range(len(test)):
        for j in range(61):
            if test[i][0][j] > 0.5:
                events[i].append(j)
    print(events)

def detect_events(network: nn.network, start_date : int):
    """Detects events from just the mobile phone network"""
    cdf = pd.read_csv("CensusDataZScore.csv", index_col = 0)
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
                            temp_array[k] -= centroids[centroid_id[j]].weekend[k]-weekend[k]
                        else:
                            temp_array[k] -= centroids[centroid_id[j]].weekday[k]-weekday[k]
                    elif k >= len(temp_array):
                        if day == 0 or day == 6:
                            temp_array = np.append(temp_array, [- centroids[centroid_id[j]].weekend[k]-weekend[k]])
                        else:
                            temp_array = np.append(temp_array, [- centroids[centroid_id[j]].weekday[k]-weekday[k]])
                    else:
                        if day == 0 or day == 6:
                             temp_array = np.insert(temp_array,k, [- centroids[centroid_id[j]].weekend[k]-weekend[k]])
                        else:
                             temp_array = np.insert(temp_array,k, [- centroids[centroid_id[j]].weekday[k]-weekday[k]])
                temp_cdf = cdf[cdf["CellID"] == j]
                temp_array = np.concatenate([temp_array,np.array([temp_cdf.iloc[0]["P1"],temp_cdf.iloc[0]["E3"],temp_cdf.iloc[0]["E4"]])])
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
                            temp_array[k] -= centroids[centroid_id[j]].weekend[k]-weekend[k]
                        else:
                            temp_array[k] -= centroids[centroid_id[j]].weekday[k]-weekday[k]
                    elif k >= len(temp_array):
                        if day == 0 or day == 6:
                            temp_array = np.append(temp_array, [- centroids[centroid_id[j]].weekend[k]-weekend[k]])
                        else:
                            temp_array = np.append(temp_array, [- centroids[centroid_id[j]].weekday[k]-weekday[k]])
                    else:
                        if day == 0 or day == 6:
                             temp_array = np.insert(temp_array,k, [- centroids[centroid_id[j]].weekend[k]-weekend[k]])
                        else:
                             temp_array = np.insert(temp_array,k, [- centroids[centroid_id[j]].weekday[k]-weekday[k]])
                temp_cdf = cdf[cdf["CellID"] == j]
                temp_array = np.concatenate([temp_array,np.array([temp_cdf.iloc[0]["P1"],temp_cdf.iloc[0]["E3"],temp_cdf.iloc[0]["E4"]])])
                prediction = network.predict(np.array([[temp_array]]))
                if prediction[0][0] > 0.5:
                    events[j].append(i)
        save_events_cell(events)
        print(i)
        
def graph_error():
    """Used to graph test error and validation error for early versions of the network"""
    validation_data = [(5638,8,12),(5638,22,12),(5638,2,11),(5638,16,12),(5638,11,12),(5638,26,12),(5638,29,11),(5638,15,12),(5638,28,12),(5638,3,11)]
    validation_actual = [[1],[1],[1],[1],[1],[0],[0],[0],[0],[0]]
    #for i in range(1,30):
    #    validation_data += [(5161,i,11),(5161,i,12)]
    #    validation_actual += [[0],[0]] 
    validation_data = load_nn_cells(validation_data)
    validation_actual = np.array(validation_actual)
    expected = [[1],[1],[1],[1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[1],[1]]
    train_data = [(5638,15,11),(5638,9,11),(5638,23,11),(5638,1,12),(5638,1,11),(5638,30,11),(5638,25,12),(5638,28,12),(5638,30,12),(6556,3,11),(6556,11,11),(6556,1,11),(6557,3,11),(6557,11,11),(6557,1,11),(6556,23,11),(6557,23,11),(6556,25,11),(6557,25,11)]
    #for i in range(1,31):
    #    expected += [[0],[0],[0],[0]]
    #    train_data += [(3788,i,11),(3789,i,11),(3688,i,11),(3689,i,11)]
    #    expected += [[0],[0],[0],[0]]
    #    train_data += [(3788,i,12),(3789,i,12),(3688,i,12),(3689,i,12)]
    #train_data += [(2142,18,11),(2143,18,11),(2144,18,11),(2042,18,11),(2043,18,11),(2044,18,11)]
    #expected += [[1],[1],[1],[1],[1],[1]]
    expected = np.array(expected)
    train_data = load_nn_cells(train_data)
    x = []
    y_train = []
    y_validation = []
    for i in range(1,101):
        net = nn.network()
        net.add(nn.fclayer(27,18))
        net.add(nn.activation(sigmoid, sigmoid_prime))
        net.add(nn.fclayer(18,3))
        net.add(nn.activation(sigmoid, sigmoid_prime))
        net.add(nn.fclayer(3,1))
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

def evaluate_cell_net(net):
    """Used to test the daily resolution mobile phone neural network"""
    struct = {}
    to_load = []
    for i in range(1,10001):
        struct[i] = []
        to_load.append(i)
    loaded = load_nn_cells_single(to_load)
    predicted = net.predict(loaded)
    for i in range(len(predicted)):
        print(i)
        for j in range(61):
            event = predicted[i][0][j]
            if event > 0.5:
                struct[i+1].append(j)
    with open("celleventssingle.pkl", "wb") as f:
        struct = pickle.dump(struct,f)

def combined_net(word_list):
    """Configures and trains the combined neural network
    
    Args
    word_list - list of words to configure the network for
    
    Returns
    Neural network object"""
    net = nn.network()
    net.add(nn.fclayer(40,24))
    net.add(nn.activation(sigmoid, sigmoid_prime))
    net.add(nn.fclayer(24,3))
    net.add(nn.activation(sigmoid, sigmoid_prime))
    net.add(nn.fclayer(3,1))
    net.add(nn.activation(sigmoid, sigmoid_prime))
    net.loss_use(nn.mse,nn.mse_prime)
    event_days = [(5638,15,11),(5638,9,11),(5638,1,12),(5638,8,12),(5638,22,12),(5638,23,11),(5638,2,11),(5638,16,12),(5638,11,12)]
    expected = [[1],[1],[1],[1],[1],[1],[1],[1],[1]]
    train_data = event_days
    test_data = []
    actual = []
    for i in range(1,31):
        test_data.append((5738,i,11))
        if not (5638, i, 11) in event_days:
            train_data += [(5638, i, 11)]
            expected += [[0]]
            actual += [[0]]
        else:
            actual += [[1]]
    for i in range(1,32):
        test_data.append((5738,i,12))
        if not (5638, i, 12) in event_days:
            train_data += [(5638, i, 12)]
            expected += [[0]]
            actual += [[0]]
        else:
            actual += [[1]]
    expected = np.array(expected)
    train_data = load_nn_combined(train_data,word_list)
    net.train(train_data,expected,10000,0.1)
    test_data = load_nn_combined(test_data,word_list)
    predicted = net.predict(test_data)
    for i in range(len(actual)):
        print("Actual : " + str(actual[i]) + " , Predicted : " + str(predicted[i][0]))
    return net

def detect_events_combined(network: nn.network, start_date : int, word_list):
    """Detect events using the combined neural network"""
    cdf = pd.read_csv("CensusDataZScore.csv", index_col = 0)
    centroids = load_centroids()
    cells = load_cells()
    cells, weekday, weekend = convert_to_residual(cells)
    centroids = attach_to_centroids(cells, centroids)
    events = load_events_combined()
    centroid_id = {}
    for i in range(len(centroids)):
        for cell in centroids[i].cells:
            centroid_id[cell.id] = i
    distro = []
    base = residual_base()
    for i in word_list:
        distro.append([])
    for i in range(61):
        print(i)
        temp = []
        for j in range(len(word_list)):
            temp.append(0)
        if i < 30:
            with open(str(i + 1).zfill(2)+"1113tweets.json","r") as f:
                tweets = tweets = json.load(f)["tweets"]
                for tweet in tweets:
                    for j in range(len(word_list)):
                        temp[j] += tweet["text"].lower().count(word_list[j])
        else:
            with open(str(i-29).zfill(2)+"1213tweets.json","r") as f:
                tweets = tweets = json.load(f)["tweets"]
                for tweet in tweets:
                    for j in range(len(word_list)):
                        temp[j] += tweet["text"].lower().count(word_list[j])
        for j in range(len(word_list)):
            distro[j].append(temp[j])
    for i in range(start_date, 61):
        if i < 30:
            df = merge_countries("2013-%s-%s" % ("11", str(i+1).zfill(2)))
            words = []
            for k in range(len(word_list)):
                words.append(distro[k][i])
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
                            temp_array[k] -= centroids[centroid_id[j]].weekend[k]-weekend[k]
                        else:
                            temp_array[k] -= centroids[centroid_id[j]].weekday[k]-weekday[k]
                    elif k >= len(temp_array):
                        if day == 0 or day == 6:
                            temp_array = np.append(temp_array, [- centroids[centroid_id[j]].weekend[k]-weekend[k]])
                        else:
                            temp_array = np.append(temp_array, [- centroids[centroid_id[j]].weekday[k]-weekday[k]])
                    else:
                        if day == 0 or day == 6:
                             temp_array = np.insert(temp_array,k, [- centroids[centroid_id[j]].weekend[k]-weekend[k]])
                        else:
                             temp_array = np.insert(temp_array,k, [- centroids[centroid_id[j]].weekday[k]-weekday[k]])
                temp_cdf = cdf[cdf["CellID"] == j]
                temp_array = np.concatenate([temp_array,np.array([temp_cdf.iloc[0]["P1"],temp_cdf.iloc[0]["E3"],temp_cdf.iloc[0]["E4"]]),np.array(words)])
                prediction = network.predict(np.array([[temp_array]]))
                if prediction[0][0] > 0.5:
                    events[j].append(i)
        else:
            df = merge_countries("2013-%s-%s" % ("12", str(i-29).zfill(2)))
            words = []
            for k in range(len(word_list)):
                words.append(distro[k][i])
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
                            temp_array[k] -= centroids[centroid_id[j]].weekend[k]-weekend[k]
                        else:
                            temp_array[k] -= centroids[centroid_id[j]].weekday[k]-weekday[k]
                    elif k >= len(temp_array):
                        if day == 0 or day == 6:
                            temp_array = np.append(temp_array, [- centroids[centroid_id[j]].weekend[k]-weekend[k]])
                        else:
                            temp_array = np.append(temp_array, [- centroids[centroid_id[j]].weekday[k]-weekday[k]])
                    else:
                        if day == 0 or day == 6:
                             temp_array = np.insert(temp_array,k, [- centroids[centroid_id[j]].weekend[k]-weekend[k]])
                        else:
                             temp_array = np.insert(temp_array,k, [- centroids[centroid_id[j]].weekday[k]-weekday[k]])
                temp_cdf = cdf[cdf["CellID"] == j]
                temp_array = np.concatenate([temp_array,np.array([temp_cdf.iloc[0]["P1"],temp_cdf.iloc[0]["E3"],temp_cdf.iloc[0]["E4"]]),np.array(words)])
                prediction = network.predict(np.array([[temp_array]]))
                if prediction[0][0] > 0.5:
                    events[j].append(i)
        save_events_combined(events)
        print(i)

def detect_events_double(cell_network: nn.network, tweet_net: nn.network, start_date : int, word_list: [str]):
    """Final implementation of event detection using both neural networks
    
    Args
    cell_network - mobile phone neural network
    tweet_net - twitter neural network
    start_date - date to start detecting events from
    word_list - words to use to detect events"""
    cdf = pd.read_csv("CensusDataZScore.csv", index_col = 0)
    centroids = load_centroids()
    cells = load_cells()
    cells, weekday, weekend = convert_to_residual(cells)
    centroids = attach_to_centroids(cells, centroids)
    events = load_events_double()
    centroid_id = {}
    for i in range(len(centroids)):
        for cell in centroids[i].cells:
            centroid_id[cell.id] = i
    distro = []
    base = residual_base()
    for i in word_list:
        distro.append([])
    for i in range(61):
        print(i)
        temp = []
        for j in range(len(word_list)):
            temp.append(0)
        if i < 30:
            with open(str(i + 1).zfill(2)+"1113tweets.json","r") as f:
                tweets = tweets = json.load(f)["tweets"]
                for tweet in tweets:
                    for j in range(len(word_list)):
                        temp[j] += tweet["text"].lower().count(word_list[j])
        else:
            with open(str(i-29).zfill(2)+"1213tweets.json","r") as f:
                tweets = tweets = json.load(f)["tweets"]
                for tweet in tweets:
                    for j in range(len(word_list)):
                        temp[j] += tweet["text"].lower().count(word_list[j])
        for j in range(len(word_list)):
            distro[j].append(temp[j])
    for i in range(start_date, 61):
        if i < 30:
            df = merge_countries("2013-%s-%s" % ("11", str(i+1).zfill(2)))
            words = []
            for k in range(len(word_list)):
                words.append(distro[k][i])
            tweet_prediction = tweet_net.predict(np.array([[words]]))
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
                            temp_array[k] -= centroids[centroid_id[j]].weekend[k]-weekend[k]
                        else:
                            temp_array[k] -= centroids[centroid_id[j]].weekday[k]-weekday[k]
                    elif k >= len(temp_array):
                        if day == 0 or day == 6:
                            temp_array = np.append(temp_array, [- centroids[centroid_id[j]].weekend[k]-weekend[k]])
                        else:
                            temp_array = np.append(temp_array, [- centroids[centroid_id[j]].weekday[k]-weekday[k]])
                    else:
                        if day == 0 or day == 6:
                             temp_array = np.insert(temp_array,k, [- centroids[centroid_id[j]].weekend[k]-weekend[k]])
                        else:
                             temp_array = np.insert(temp_array,k, [- centroids[centroid_id[j]].weekday[k]-weekday[k]])
                temp_cdf = cdf[cdf["CellID"] == j]
                temp_array = np.concatenate([temp_array,np.array([temp_cdf.iloc[0]["P1"],temp_cdf.iloc[0]["E3"],temp_cdf.iloc[0]["E4"]])])
                cell_prediction = cell_network.predict(np.array([[temp_array]]))
                if cell_prediction[0][0] > 0.5 and tweet_prediction[0][0] > 0.5:
                    events[j].append(i)
        else:
            df = merge_countries("2013-%s-%s" % ("12", str(i-29).zfill(2)))
            words = []
            for k in range(len(word_list)):
                words.append(distro[k][i])
            tweet_prediction = tweet_net.predict(np.array([[words]]))
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
                            temp_array[k] -= centroids[centroid_id[j]].weekend[k]-weekend[k]
                        else:
                            temp_array[k] -= centroids[centroid_id[j]].weekday[k]-weekday[k]
                    elif k >= len(temp_array):
                        if day == 0 or day == 6:
                            temp_array = np.append(temp_array, [- centroids[centroid_id[j]].weekend[k]-weekend[k]])
                        else:
                            temp_array = np.append(temp_array, [- centroids[centroid_id[j]].weekday[k]-weekday[k]])
                    else:
                        if day == 0 or day == 6:
                             temp_array = np.insert(temp_array,k, [- centroids[centroid_id[j]].weekend[k]-weekend[k]])
                        else:
                             temp_array = np.insert(temp_array,k, [- centroids[centroid_id[j]].weekday[k]-weekday[k]])
                temp_cdf = cdf[cdf["CellID"] == j]
                temp_array = np.concatenate([temp_array,np.array([temp_cdf.iloc[0]["P1"],temp_cdf.iloc[0]["E3"],temp_cdf.iloc[0]["E4"]])])
                cell_prediction = cell_network.predict(np.array([[temp_array]]))
                if cell_prediction[0][0] > 0.5 and tweet_prediction[0][0] > 0.5:
                    events[j].append(i)
        save_events_double(events)
        print(i)

def save_events_cell(struct: dict):
    """Saves mobile phone detected events
    
    Args
    struct - detected events to save"""
    with open("cellevents.pkl","wb") as f:
        pickle.dump(struct, f)

def save_events_combined(struct: dict):
    """Saves combined nn detected events
    
    Args
    struct - detected events to save"""
    with open("combinedevents.pkl","wb") as f:
        pickle.dump(struct, f)

def save_events_double(struct: dict):
    """Saves events detected by both nn
    
    Args
    struct - detected events to save"""
    with open("doubleevents.pkl","wb") as f:
        pickle.dump(struct, f)

def load_events_cell():
    """Loads detected events
    
    Args
    returns dictionary with detected events"""
    with open("cellevents.pkl", "rb") as f:
        struct = pickle.load(f)
    return struct

def load_events_combined():
    """Loads detected events
    
    Args
    returns dictionary with detected events"""
    with open("combinedevents.pkl", "rb") as f:
        struct = pickle.load(f)
    return struct

def load_events_double():
    """Loads detected events
    
    Args
    returns dictionary with detected events"""
    with open("doubleevents.pkl", "rb") as f:
        struct = pickle.load(f)
    return struct
        
def save_network_cell(struct: nn.network):
    """Saves a neural network"""
    with open("cellnetwork.pkl","wb") as f:
        pickle.dump(struct, f)

def save_network_combined(struct: nn.network):
    """Saves a neural network"""
    with open("combinednetwork.pkl","wb") as f:
        pickle.dump(struct, f)

def save_network_cell_single(struct: nn.network):
    """Saves a neural network"""
    with open("cellnetworksingle.pkl","wb") as f:
        pickle.dump(struct, f)

def load_network_cell():
    """Loads a neural network"""
    with open("cellnetwork.pkl", "rb") as f:
        struct = pickle.load(f)
    return struct

def load_network_combined():
    """Loads a neural network"""
    with open("combinednetwork.pkl", "rb") as f:
        struct = pickle.load(f)
    return struct

def load_network_cell_single():
    """Loads a neural network"""
    with open("cellnetworksingle.pkl", "rb") as f:
        struct = pickle.load(f)
    return struct

def save_network_tweet(struct: nn.network):
    """Saves a neural network"""
    with open("tweetnetwork.pkl","wb") as f:
        pickle.dump(struct, f)

def load_network_tweet():
    """Loads a neural network"""
    with open("tweetnetwork.pkl", "rb") as f:
        struct = pickle.load(f)
    return struct

if __name__ == "__main__":
    print("start")
    #net = cell_network()
    #save_network_cell(net)
    #net = load_network_cell_single()
    #net = cell_network_san_siro_only()
    #net = save_network_cell(net)
    #struct = {}
    #for i in range(1,10001):
    #    struct[i] = []
    #save_events_double(struct)
    #net = load_network_cell()
    #detect_events(net, 0)
    #struct = load_events_cell()
    #print(struct)
    #graph_error()
    #net = tweet_network_single_word()
    #print(load_nn_combined([(5638,15,11)],["inter","parma","sampdoria","livorno","ajax","roma","fiorentina","genoa","arctic monkeys", "pixies", "skrillex","bastille", "bring me the horizon"]))
    word_list = ["inter", "ac milan","parma","sampdoria","livorno","ajax","roma","fiorentina","genoa","arctic monkeys"]
    tweet_net = tweet_network(word_list)
    #cell_net = cell_network_san_siro_only()
    #cell_net = load_network_cell()
    #tweet_net = load_network_tweet()
    #net = combined_net(word_list)
    #save_network_combined(net)
    #net = load_network_combined()
    #detect_events_double(cell_net,tweet_net,0,word_list)