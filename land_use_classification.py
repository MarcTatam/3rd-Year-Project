from datetime import datetime
from random import uniform
import numpy as np
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
    """Assigns each cell to a centroid
    
    Args
    cells - List of the cells
    centroids - List of the centroids
    
    Returns
    List of the centroids with the cells attached"""
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
    """Positions the centroid within the cluster
    
    Args
    centroids - list of centroids with the cells attached
    
    Returns
    List of centroids postioned within the cell"""
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

def k_means(iterations : int, centroids: int, cells: [Cell], minn:float, maxx :float)->[Centroid]:
    """Performs the K-means algorithm
    
    Args
    iterations - Number of iterations to perform
    centroids - Number of centroids to use
    cells - List of cells to use
    minn - Minimum value for the cells (used for centroid initilisation)
    maxx - Maximum value for the cells (used for centroid initilisation)
    
    Returns
    List of centroids"""
    centroid_list = []
    for i in range(centroids):
        this_centroid = Centroid()
        for j in range(24):
            this_centroid.weekday[j] = uniform(minn, maxx)
            this_centroid.weekend[j] = uniform(minn, maxx)
        centroid_list.append(this_centroid)
    for i in range(iterations):
        centroid_list = attach_to_centroids(cells, centroid_list)
        centroid_list = position_centroids(centroid_list)
        print(i)
    return centroid_list


def save_cells(cell_list: [Cell]):
    """Saves the cells to a JSON file
    
    Args
    cell_list - list of cells to save"""
    cells = []
    for cell in cell_list:
        cells.append(cell.to_json())
    with open("cells.json","w") as f:
        json.dump({"cells" : cells},f)

def load_cells()->[Cell]:
    """Loads the cells from a JSON file
    
    Returns
    List of loaded cells"""
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
    """Saves the centroids to a JSON file
    
    Args
    centroids - list of centroids to save"""
    centroid_list = []
    for centroid in centroids:
        centroid_list.append(centroid.to_json())
    with open("centroids.json","w") as f:
        json.dump({"centroids" : centroid_list},f)

def load_centroids()->[Centroid]:
    """Loads the centroids from a JSON file
    
    Returns
    List of loaded centroids"""
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
    """Parses data from original csv files to a single hourly csv file"""
    data_list = []
    for cellid in range(1,10001):
        data_list.append(Cell(cellid))
    cdr = ocdr.merge_all()
    cdr[8] = (pd.to_datetime(cdr[1],unit='ms')+dt.timedelta(hours = 1)).dt.strftime("%w")
    cdr[9] = (pd.to_datetime(cdr[1],unit='ms')+dt.timedelta(hours = 1)).dt.strftime("%H")
    cdr = cdr.groupby([0,8,9]).sum()
    cdr.to_csv("merged.csv")

def open_data()->pd.DataFrame:
    """Opens parsed data
    
    Returns
    Dataframe of parsed data"""
    cdr = pd.read_csv("merged.csv")
    return cdr

def remove_excess(cdr:pd.DataFrame)->pd.DataFrame:
    """Removes unwanted columns (DEPRECATED)
    
    Args
    cdr - untrimmed dataframe
    
    Returns
    Trimmed dataframe with excess columns removed"""
    return cdr.drop('1', axis = 'columns')

def sort_daytype(cdr:pd.DataFrame)->pd.DataFrame:
    """Sorts the data into each day type.
    1 represents weekends and 0 represents weekdays
    
    Args
    cdr - data to sort into daytype"""
    def daytype(arg):
        if arg == 0 or arg == 6:
            return 1
        else:
            return 0
    cdr["8"] = cdr["8"].apply(daytype)
    return cdr

def min_max_normalise(df:pd.DataFrame, column:str)-> pd.DataFrame:
    """Normalises the data using min max normilisation
    
    Args
    df - Dataframe to normalise
    column - column to normalise
    
    Returns
    Dataframe with desired column normalised"""
    minn = df[column].min()
    maxx = df[column].max()
    df[column] = (df[column]-minn)/(maxx-minn)
    return df

def format_data(weekday:pd.DataFrame, weekend: pd.DataFrame)->[Cell]:
    """Converts the data from Dataframes to sells
    
    Args
    weekday - Dataframe containing all the weeday activities
    weekend - Dataframe containing all the weekend activities
    
    Returns
    List of cells"""
    cells = []
    for i in range(1, 10001):
        cell = Cell(i)
        cell.weekday = weekday.loc[weekday["0"] == i]["7"].to_list()
        cell.weekend = weekend.loc[weekend["0"] == i]["7"].to_list()
        cells.append(cell)
    return cells


def format_data_text(weekday:pd.DataFrame, weekend: pd.DataFrame)->[Cell]:
    """Converts the data from Dataframes to sells
    
    Args
    weekday - Dataframe containing all the weeday activities
    weekend - Dataframe containing all the weekend activities
    
    Returns
    List of cells"""
    cells = []
    weekday["7"] = weekday["3"] + weekday["4"]
    weekend["7"] = weekend["3"] + weekend["4"]
    for i in range(1, 10001):
        cell = Cell(i)
        cell.weekday = weekday.loc[weekday["0"] == i]["7"].to_list()
        cell.weekend = weekend.loc[weekend["0"] == i]["7"].to_list()
        cells.append(cell)
    return cells

def format_data_all(weekday:pd.DataFrame, weekend: pd.DataFrame)->[Cell]:
    """Converts the data from Dataframes to sells
    
    Args
    weekday - Dataframe containing all the weeday activities
    weekend - Dataframe containing all the weekend activities
    
    Returns
    List of cells"""
    cells = []
    weekday["7"] = weekday["3"] + weekday["4"] + weekday["5"] + weekday["6"]+ weekday["7"]
    weekend["7"] = weekend["3"] + weekend["4"] + weekend["5"] + weekend["6"]+ weekend["7"]
    for i in range(1, 10001):
        cell = Cell(i)
        cell.weekday = weekday.loc[weekday["0"] == i]["7"].to_list()
        cell.weekend = weekend.loc[weekend["0"] == i]["7"].to_list()
        cells.append(cell)
    return cells

def format_data_pruned(weekday:pd.DataFrame, weekend: pd.DataFrame)->[Cell]:
    """Formats Data if only looking at a subset of cells
    
    Args
    weekday - Dataframe containing all the weeday activities
    weekend - Dataframe containing all the weekend activities
    
    Returns
    List of cells"""
    cells = []
    for i in range(1, 10001):
        print(i)
        if i >= 3100 and i <= 8400 and i % 100 >= 20 and i % 100 <= 80:
            cell = Cell(i)
            cell.weekday = weekday.loc[weekday["0"] == i]["7"].to_list()
            cell.weekend = weekend.loc[weekend["0"] == i]["7"].to_list()
            cells.append(cell)
    return cells

def prune_unwanted(df:pd.DataFrame)->pd.DataFrame:
    """Prunes data for when looking at a specific subset of cells
    
    Args
    df - Dataframe to prune unwanted data
    
    Returns dataframe with subset of cells"""
    df = df[(df["0"] >= 3100) & (df["0"] <= 8400)]
    df = df[(df["0"]%100 >= 20) & (df["0"]%100 <= 80)]
    return df

def zscore_normalise(cells:[Cell])->[Cell]:
    """Normalises cells using z score normilisation
    
    Args
    cells - List of cells to be normalised
    
    Returns
    List of cells with their data normalised"""
    for cell in cells:
        mean = np.mean(cell.weekday)
        print(mean)
        sd = np.std(cell.weekday)
        for i in range(len(cell.weekday)):
            cell.weekday[i] = (cell.weekday[i]-mean)/sd
        mean = np.mean(cell.weekend)
        sd = np.std(cell.weekend)
        for i in range(len(cell.weekend)):
            cell.weekend[i] = (cell.weekend[i]-mean)/sd
    return cells

def get_min_max(cells:[Cell])->(float,float):
    """Gets minimum and maximum values
    
    Args
    cells - list of cells to find the minimum and maximum values from
    
    Returns
    Minimum and maximum values for the dataset"""
    minn = None
    maxx = None
    for cell in cells:
        if minn == None:
            minn = min(cell.weekend+cell.weekday)
            maxx = max(cell.weekend+cell.weekday)
        else:
            minn = min(cell.weekend+cell.weekday+[minn])
            maxx = max(cell.weekend+cell.weekday+[maxx])
    return (minn,maxx)

def convert_to_residual(cells:[Cell])->([Cell],[float],[float]):
    """Converts the cells from normalised form to residual data
    
    Args
    cells - list of cells to convert to normalised form
    
    Returns
    Normalised cells, average weekday activity, average weekend activity"""
    weekend = [0]*24
    weekday = [0]*24
    for cell in cells:
        for i in range(24):
            weekend[i] += cell.weekend[i]
            weekday[i] += cell.weekday[i]
    weekday = list(map(lambda x: x/10000, weekday))
    weekend = list(map(lambda x: x/10000, weekend))
    for cell in cells:
        for i in range(24):
            cell.weekend[i] = cell.weekend[i] - weekend[i]
            cell.weekday[i] = cell.weekday[i] - weekday[i]
    return cells, weekday, weekend

def centroids_daily_pattern():
    """Gets the daily centroid zscore and saves them"""
    def project_time(day, month):
        if int(month) == 11:
            return int(day)-1
        else:
            return int(day)+29
    print("start")
    cells = load_cells()
    centroids = load_centroids()
    centroids = attach_to_centroids(cells, centroids)
    df = pd.read_csv("loose_merge.csv", index_col=0)
    struct = {"centroids" : []}
    for centroid in centroids:
        centroid_distro = [0]*61
        for cell in centroid.cells:
            print(cell.id)
            temp_df = df[df["0"]==cell.id]
            temp_df["8"] = (pd.to_datetime(temp_df["1"],unit='ms')+dt.timedelta(hours = 1)).dt.strftime("%d")
            temp_df["9"] = (pd.to_datetime(temp_df["1"],unit='ms')+dt.timedelta(hours = 1)).dt.strftime("%m")
            temp_df = temp_df.groupby(["9","8"]).sum().reset_index()
            temp_df["10"] = temp_df.apply(lambda x: project_time(x["8"],x["9"]),axis = 1)
            value_list = temp_df["7"].tolist()
            days_list = temp_df["10"].tolist()
            for i in range(len(days_list)):
                centroid_distro[days_list[i]] += value_list[i]
        centroid_distro = np.array(centroid_distro)
        std = centroid_distro.std()
        mean = centroid_distro.mean()
        centroid_distro = (centroid_distro - mean)/std
        struct["centroids"].append({"pattern" : centroid_distro.tolist()})
    with open("centroid_pattern.json","w+") as f:
        json.dump(struct, f)


            

if __name__ == "__main__":
    #parse_data()
    #cdr = open_data()
    #cdr = sort_daytype(cdr)
    #cdr = cdr.groupby(["0","8","9"]).sum().reset_index()
    #print(cdr)
    #cells = format_data_all(cdr[cdr["8"]==0], cdr[cdr["8"]==1])
    #cells = zscore_normalise(cells)
    #save_cells(cells)
    #print(len(cells[0].weekend))
    #cells = load_cells()
    #cells = convert_to_residual(cells)[0]
    #minn,maxx = get_min_max(cells)
    #centroids = k_means(1000, 5, cells, minn, maxx)
    #save_centroids(centroids)
    centroids_daily_pattern()
    

