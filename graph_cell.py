from land_use_classification import open_data, load_cells, load_centroids, convert_to_residual, attach_to_centroids
from open_cdr import merge_all
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import project_utils as util

def save_merged():
    """Saves merged files"""
    df = merge_all()
    df.to_csv("loose_merge.csv")

def load_merged():
    """Loads merged files"""
    df = pd.read_csv("loose_merge.csv")
    del df["Unnamed: 0"]
    return df

def get_centroid_ind(centroids, target):
    """Gets the centroid which a cell is attached to
    
    Args
    centroids - list of centroids
    target - cell to find attachment too"""
    for i in range(len(centroids)):
        for cell_obj in centroids[i].cells:
            if cell_obj.id -1 == target:
                return i

def graph_cell(cell:int):
    """Graphs the activity of a cell
    
    Args
    cell - cell to plot activity for"""
    def daytype(arg):
        if arg == 0 or arg == 6:
            return 1
        else:
            return 0
    df = load_merged()
    centroids = load_centroids()
    cells = load_cells()
    cells, weekday, weekend = convert_to_residual(cells)
    centroids = attach_to_centroids(cells, centroids)
    wanted_centroid = get_centroid_ind(centroids, cell)
    df = df[df["0"] == cell]
    df["8"] = (pd.to_datetime(df["1"],unit='ms')+dt.timedelta(hours = 1)).dt.strftime("%w")
    df["10"] = (pd.to_datetime(df["1"],unit='ms')+dt.timedelta(hours = 1)).dt.strftime("%d")
    df["11"] = (pd.to_datetime(df["1"],unit='ms')+dt.timedelta(hours = 1)).dt.strftime("%m")
    df["1"] = (pd.to_datetime(df["1"],unit='ms')+dt.timedelta(hours = 1)).dt.strftime("%H")
    df = df.groupby(["11","10","8","1"]).sum().reset_index()
    df["8"] = df["8"].apply(daytype)
    df["7"]=(df["7"] - df["7"].mean())/df["7"].std(ddof=0)
    x = []
    for i in range(1464):
        x.append(i)
    y = []
    for index, row in df.iterrows():
        print(row)
        if row["8"] == 1:
            y.append(row["7"]-centroids[wanted_centroid].weekend[int(row["1"])]-weekend[int(row["1"])])
        else:
            y.append(row["7"]-centroids[wanted_centroid].weekday[int(row["1"])]-weekday[int(row["1"])])
    fig, ax = plt.subplots()
    ax.plot(x,y)
    plt.show()

def graph_cells(cell_ids:[int], events, filename):
    """Graphs the activity of multiple cells, highlighting events
    
    Args
    cell_ids - cells to plot activity for
    events - events to highlight by name
    filename - filename to save graph to"""
    def daytype(arg):
        if arg == 0 or arg == 6:
            return 1
        else:
            return 0
    x = []
    for i in range(1464):
        x.append(i)
    centroids = load_centroids()
    cells = load_cells()
    cells, weekday, weekend = convert_to_residual(cells)
    centroids = attach_to_centroids(cells, centroids)
    df = load_merged()
    fig, ax = plt.subplots()
    lines = []
    for cell in cell_ids:
        print(cell)
        wanted_centroid = get_centroid_ind(centroids, cell)
        temp_df = df[df["0"] == cell]
        temp_df["8"] = (pd.to_datetime(temp_df["1"],unit='ms')+dt.timedelta(hours = 1)).dt.strftime("%w")
        temp_df["10"] = (pd.to_datetime(temp_df["1"],unit='ms')+dt.timedelta(hours = 1)).dt.strftime("%d")
        temp_df["11"] = (pd.to_datetime(temp_df["1"],unit='ms')+dt.timedelta(hours = 1)).dt.strftime("%m")
        temp_df["1"] = (pd.to_datetime(temp_df["1"],unit='ms')+dt.timedelta(hours = 1)).dt.strftime("%H")
        temp_df = temp_df.groupby(["11","10","8","1"]).sum().reset_index()
        temp_df["8"] = temp_df["8"].apply(daytype)
        temp_df["7"]=(temp_df["7"] - temp_df["7"].mean())/temp_df["7"].std(ddof=0)
        print(temp_df)
        x = []
        y = []
        for index, row in temp_df.iterrows():
            if row["8"] == 1:
                y.append(row["7"]-centroids[wanted_centroid].weekend[int(row["1"])]-weekend[int(row["1"])])
            else:
                y.append(row["7"]-centroids[wanted_centroid].weekday[int(row["1"])]-weekday[int(row["1"])])
        for i in range(len(y)):
            x.append(i)
        line, = ax.plot(x,y)
        line.set_label(cell)
    event_df = pd.read_csv("Events.csv")
    for event in events:
        for index, row in event_df.loc[event_df["Name"] == event].iterrows():
            start,stop = util.convert_to_project_hourly(row["Date"])
            ax.axvspan(start,stop, color = "silver")
    ax.set_ylabel("Residual based on centroid average")
    ax.set_xlabel("Time in hours since project epoch")
    ax.set_title("Residual activity of the cells containing Alcatraz")
    ax.legend(loc="upper right",prop={'size': 6})
    #plt.show()
    plt.savefig(filename+".png")

def graph_cells_epoch(cell_ids:[int], events, filename):
    """Graphs the activity of multiple cells, highlighting events
    
    Args
    cell_ids - cells to plot activity for
    events - events to highlight in days from project epoch
    filename - filename to save graph to"""
    def daytype(arg):
        if arg == 0 or arg == 6:
            return 1
        else:
            return 0
    x = []
    for i in range(1464):
        x.append(i)
    centroids = load_centroids()
    cells = load_cells()
    cells, weekday, weekend = convert_to_residual(cells)
    centroids = attach_to_centroids(cells, centroids)
    df = load_merged()
    fig, ax = plt.subplots()
    lines = []
    for cell in cell_ids:
        print(cell)
        wanted_centroid = get_centroid_ind(centroids, cell)
        temp_df = df[df["0"] == cell]
        temp_df["8"] = (pd.to_datetime(temp_df["1"],unit='ms')+dt.timedelta(hours = 1)).dt.strftime("%w")
        temp_df["10"] = (pd.to_datetime(temp_df["1"],unit='ms')+dt.timedelta(hours = 1)).dt.strftime("%d")
        temp_df["11"] = (pd.to_datetime(temp_df["1"],unit='ms')+dt.timedelta(hours = 1)).dt.strftime("%m")
        temp_df["1"] = (pd.to_datetime(temp_df["1"],unit='ms')+dt.timedelta(hours = 1)).dt.strftime("%H")
        temp_df = temp_df.groupby(["11","10","8","1"]).sum().reset_index()
        temp_df["8"] = temp_df["8"].apply(daytype)
        temp_df["7"]=(temp_df["7"] - temp_df["7"].mean())/temp_df["7"].std(ddof=0)
        print(temp_df)
        x = []
        y = []
        for index, row in temp_df.iterrows():
            if row["8"] == 1:
                y.append(row["7"]-centroids[wanted_centroid].weekend[int(row["1"])]-weekend[int(row["1"])])
            else:
                y.append(row["7"]-centroids[wanted_centroid].weekday[int(row["1"])]-weekday[int(row["1"])])
        for i in range(len(y)):
            x.append(i)
        line, = ax.plot(x,y)
        line.set_label(cell)
    event_df = pd.read_csv("Events.csv")
    for event in events:
        start = event*24
        stop = event*24 + 23
        ax.axvspan(start,stop, color = "silver")
    ax.set_ylabel("Residual based on centroid average")
    ax.set_xlabel("Time in hours since project epoch")
    ax.legend(loc="upper right",prop={'size': 6})
    #plt.show()
    plt.savefig(filename+".png")

if __name__ == "__main__":
    graph_cells_epoch([1], [1,2,4,8,16], "demonstration")