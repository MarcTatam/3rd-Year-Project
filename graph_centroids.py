import matplotlib.pyplot as plt
from land_use_classification import load_centroids
from math import log

def graph_log_weekday():
    centroids = load_centroids()
    x = []
    for i in range(24):
        x.append(i)
    fig, ax = plt.subplots()
    for centroid in centroids:
        ax.plot(x,list(map(log, centroid.weekday)))
    plt.show()

def graph_weekday():
    centroids = load_centroids()
    x = []
    for i in range(24):
        x.append(i)
    fig, ax = plt.subplots()
    for centroid in centroids:
        ax.plot(x,centroid.weekday)
    plt.show()

def graph_log_weekend():
    centroids = load_centroids()
    x = []
    for i in range(24):
        x.append(i)
    fig, ax = plt.subplots()
    for centroid in centroids:
        ax.plot(x,list(map(log, centroid.weekend)))
    plt.show()

def graph_weekend():
    centroids = load_centroids()
    x = []
    for i in range(24):
        x.append(i)
    fig, ax = plt.subplots()
    for centroid in centroids:
        ax.plot(x,centroid.weekend)
    plt.show()

if __name__ == "__main__":
    graph_weekend()
