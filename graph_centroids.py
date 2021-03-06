import matplotlib.pyplot as plt
from land_use_classification import load_centroids, load_cells, convert_to_residual
import numpy as np
from math import log

def graph_log_weekday():
    """Graphs the log activity of the cluster on a weekday using an hourly timescale"""
    centroids = load_centroids()
    x = []
    for i in range(24):
        x.append(i)
    fig, ax = plt.subplots()
    for centroid in centroids:
        ax.plot(x,list(map(log, centroid.weekday)))
    plt.show()

def graph_weekday():
    """Graphs the activity of the cluster on a weekday using an hourly timescale"""
    centroids = load_centroids()
    cells = load_cells()
    cells, weekday, weekend = convert_to_residual(cells)
    x = []
    for i in range(24):
        x.append(i)
    fig, ax = plt.subplots()
    i = 1
    for centroid in centroids:
        to_plot = []
        for j in range(24):
            to_plot.append(weekday[j] - centroid.weekday[j])
        ax.plot(x,to_plot, label = i)
        i+=1
    fig.legend(loc="upper right", title = "Centroid")
    ax.set_title("Weekday Activity")
    ax.set_ylabel("Activity")
    ax.set_xlabel("Hour of the Day")
    #plt.show()
    plt.savefig('Weekday3.png')

def graph_log_weekend():
    """Graphs the log activity of the cluster on a weekend using an hourly timescale"""
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
    """Graphs the activity of the cluster on a weekend using an hourly timescale"""
    centroids = load_centroids()
    cells = load_cells()
    cells, weekday, weekend = convert_to_residual(cells)
    x = []
    for i in range(24):
        x.append(i)
    fig, ax = plt.subplots()
    i=1
    for centroid in centroids:
        to_plot = []
        for j in range(24):
            to_plot.append(weekend[j] - centroid.weekend[j])
        ax.plot(x,to_plot, label = i)
        i+=1
    fig.legend(loc="upper right", title = "Centroid")
    ax.set_title("Weekend Activity")
    ax.set_ylabel("Activity")
    ax.set_xlabel("Hour of the Day")
    #plt.show()
    plt.savefig('Weekend3.png')

def graph_centroids():
    """Graphs the activity of the cluster in a single graph using a hourly timescale"""
    centroids = load_centroids()
    cells = load_cells()
    cells, weekday, weekend = convert_to_residual(cells)
    x = []
    color_list = ["darkgreen","purple","hotpink","red","orange"]
    for i in range(48):
        x.append(i)
    fig, ax = plt.subplots()
    i=1
    for centroid in centroids:
        to_plot = []
        for j in range(24):
            to_plot.append(weekday[j] + centroid.weekday[j])
        for j in range(24):
            to_plot.append(weekend[j] + centroid.weekend[j])
        ax.plot(x,to_plot, label = i, color=color_list[i-1])
        i+=1
    fig.legend(loc="upper right", title = "Centroid")
    ax.set_title("Centroid Activity")
    ax.set_ylabel("Normalised Activity")
    ax.set_xlabel("Hour of the Day")
    ax.xaxis.set_ticklabels([])
    ax.xaxis.set_ticks(np.arange(0, 47, 4))
    ax.axvline(24,color = "black", linestyle = "--")
    #plt.show()
    plt.savefig('Centroids.png')
if __name__ == "__main__":
    graph_centroids()
