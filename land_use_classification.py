from datetime import datetime
from random import random
import pandas as pd
import datetime as dt
import open_cdr as ocdr
import json
import math

class Cell:
    """Class representing a cell"""
    def __init__(self, id:int):
        """Constructor
        
        Args
        self - instance identifier
        id - Cell id"""
        self.weekend = []
        self.weekday = []
        self.id = id

    def to_json(self)->dict:
        """Converts the cell to a JSON serializable object
        
        Args
        self - instance identifier
        
        Returns
        Object which can be serialized into a json file"""
        return {"weekend" : self.weekend, "weekday" : self.weekday, "id" : self.id}

    def distance_to(self, centroid)->float:
        """Calculates the distance to a centroid
        
        Args
        self - instance identifier
        centroid - the centroid to calculate the distance to
        
        Returns
        Float for the distance to a value"""
        temp_sum = 0
        for i in range(24):
            temp_sum = (self.weekend[i] - centroid.weekend[i])**2
            temp_sum = (self.weekday[i] - centroid.weekday[i])**2
        return math.sqrt(temp_sum)

class Centroid:
    """Class representing a centroid"""
    def __init__(self):
        """Constructor"""
        self.weekend = [0]*24
        self.weekday = [0]*24
        self.cells = []

    def add_cell(self, cell):
        """Adds a cell to the list of those attached to the centroid
        
        Args
        self - instance identifier
        cell - cell to attach to the centroid"""
        self.cells.append(cell)

    def to_json(self)->dict:
        """Converts the centroid to a JSON serializable object
        
        Args
        self - instance identifier
        
        Returns
        Object which can be serialized into a json file"""
        return {"weekend" : self.weekend, "weekday" : self.weekday}

def attach_to_centroids(cells:[Cell],centroids:[Centroid])->[Centroid]:
    #Clear centroids of attachments
    for centroid in centroids:
        centroid.cells = []
    for cell in cells:
        nearest = None
        distance = None
        for i in range(len(centroids)):
            distance_to = cell.distance_to(centroids[i])
            if distance == None:
                nearest = i
                distance = distance_to
            elif distance_to < distance:
                nearest = i
                distance = distance_to
        centroids[nearest].add_cell(cell)
    return centroids

def position_centroids(centroids:[Centroid])->[Centroid]:
    for i in range(len(centroids)):
        #First sum the positions of each cell
        sum_weekday = [0]*24
        sum_weekend = [0]*24
        for cell in centroids[i].cells:
            for j in range(24):
                sum_weekday[j] += cell.weekday[j]
                sum_weekend[j] += cell.weekend[j]
        #Then find average position
        if len(centroids[i].cells) > 0:
            for j in range(24):
                sum_weekend[j] = sum_weekend[j]/len(centroids[i].cells)
                sum_weekday[j] = sum_weekday[j]/len(centroids[i].cells)
            centroids[i].weekend = sum_weekend
            centroids[i].weekday = sum_weekday
    return centroids

def k_means(iterations : int, centroids: int, cells: [Cell]):
    centroid_list = []
    for i in range(centroids):
        this_centroid = Centroid()
        for j in range(24):
            this_centroid.weekday[j] = random()
            this_centroid.weekend[j] = random()
        centroid_list.append(this_centroid)
    for i in range(iterations):
        print(i)
        print(centroid_list)
        centroid_list = attach_to_centroids(cells, centroid_list)
        centroid_list = position_centroids(centroid_list)
    return centroid_list


def save_cells(cell_list: [Cell]):
    cells = []
    for cell in cell_list:
        cells.append(cell.to_json())
    with open("cells.json","w") as f:
        json.dump({"cells" : cells},f)

def load_cells()->[Cell]:
    with open("cells.json","r+") as f:
        json_cells = json.load(f)["cells"]
    cells = []
    for item in json_cells:
        cell = Cell(item["id"])
        cell.weekday = item["weekday"]
        cell.weekend = item["weekend"]
        cells.append(cell)
    return cells

def save_centroids(centroids: [Centroid]):
    centroid_list = []
    for centroid in centroids:
        centroid_list.append(centroid.to_json())
    with open("centroids.json","w") as f:
        json.dump({"centroids" : centroid_list},f)

def load_centroids()->[Centroid]:
    with open("centroids.json","r+") as f:
        json_centroids = json.load(f)["centroids"]
    centroids = []
    for item in json_centroids:
        centroid = Centroid()
        centroid.weekday = item["weekday"]
        centroid.weekend = item["weekend"]
        centroids.append(centroid)
    return centroids

def parse_data():
    data_list = []
    for cellid in range(1,10001):
        data_list.append(Cell(cellid))
    cdr = ocdr.merge_all()
    #cdr = ocdr.merge_countries("2013-11-01")
    cdr[8] = (pd.to_datetime(cdr[1],unit='ms')+dt.timedelta(hours = 1)).dt.strftime("%w")
    cdr[9] = (pd.to_datetime(cdr[1],unit='ms')+dt.timedelta(hours = 1)).dt.strftime("%H")
    cdr = cdr.groupby([0,8,9]).sum()
    cdr.to_csv("merged.csv")

def open_data():
    cdr = pd.read_csv("merged.csv")
    return cdr

def remove_excess(cdr:pd.DataFrame)->pd.DataFrame:
    return cdr.drop('1', axis = 'columns')

def sort_daytype(cdr:pd.DataFrame)->pd.DataFrame:
    def daytype(arg):
        if arg == 0 or arg == 6:
            return 1
        else:
            return 0
    cdr["8"] = cdr["8"].apply(daytype)
    return cdr

def normalise(df:pd.DataFrame, column:str)-> pd.DataFrame:
    minn = df[column].min()
    maxx = df[column].max()
    df[column] = (df[column]-minn)/(maxx-minn)
    return df

def format_data(weekday:pd.DataFrame, weekend: pd.DataFrame)->[Cell]:
    cells = []
    for i in range(1, 10001):
        print(i)
        cell = Cell(i)
        cell.weekday = weekday.loc[weekday["0"] == i]["7"].to_list()
        cell.weekend = weekend.loc[weekend["0"] == i]["7"].to_list()
        cells.append(cell)
    print(cells)
    return cells

def prune_unwanted(df:pd.DataFrame)->pd.DataFrame:
    pass

if __name__ == "__main__":
    cdr = open_data()
    print(cdr)
    #cdr["7"] = cdr["7"]**(1/3)
    #cdr = normalise(cdr, "7")
    #cdr = sort_daytype(cdr)
    #cells = format_data(cdr[cdr["8"]==0], cdr[cdr["8"]==0])
    #save_cells(cells)
    #cells = load_cells()
    #centroids = k_means(1000, 5, cells)
    #save_centroids(centroids)
    

