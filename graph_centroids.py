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
    i = 1
    for centroid in centroids:
        ax.plot(x,centroid.weekday, label=i)
        i += 1
    fig.legend(loc="upper right", title = "Centroid")
    ax.set_title("Weekday Activity")
    ax.set_ylabel("Activity")
    ax.set_xlabel("Hour of the Day")
    #plt.show()
    plt.savefig('Weekday.png')

def graph_log_weekend():
    centroids = load_centroids()
    x = []
    for i in range(24):
        x.append(i)
    fig, ax = plt.subplots()
    lines = []
    for centroid in centroids:
        line = ax.plot(x,list(map(log, centroid.weekend)))
        lines.append(line)
    ax.legend(lines,[1,2,3,4,5])
    plt.show()

def graph_weekend():
    centroids = load_centroids()
    x = []
    for i in range(24):
        x.append(i)
    fig, ax = plt.subplots()
    i=1
    for centroid in centroids:
        ax.plot(x,centroid.weekend, label = i)
        i+=1
    fig.legend(loc="upper right", title = "Centroid")
    ax.set_title("Weekend Activity")
    ax.set_ylabel("Activity")
    ax.set_xlabel("Hour of the Day")
    #plt.show()
    plt.savefig('Weekend.png')
if __name__ == "__main__":
    graph_weekend()
