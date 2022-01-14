from datetime import datetime
import pandas as pd
import datetime as dt
import open_cdr as ocdr
import json
import math

class Cell:
    """representing a cell"""
    def __init__(self, id:int):
        self.weekend = []
        self.weekday = []
        self.id = id

    def to_json(self):
        return {"weekend" : self.weekend, "weekday" : self.weekday, "id" : self.id}

    def distance_to(self, centroid)->float:
        temp_sum = 0
        for i in range(23):
            temp_sum = (self.weekend[i] - centroid.weekend[i])**2
            temp_sum = (self.weekday[i] - centroid.weekday[i])**2
        return math.sqrt(temp_sum)

class Centroid:


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
    return cell

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

if __name__ == "__main__":
    cell_list = [Cell(1)]
    cdr = open_data()
    cdr = remove_excess(cdr)
    cdr = sort_daytype(cdr)
    cdr = cdr.groupby(["0","8","9"]).sum()
    cdr = cdr.reset_index()
    weekday = cdr.loc[cdr['8'] == 0]
    weekend = cdr.loc[cdr['8'] == 1]
    weekday = normalise(weekday, "7")
    weekend = normalise(weekend, "7")
    cells = format_data(weekday, weekend)
    save_cells(cells)
