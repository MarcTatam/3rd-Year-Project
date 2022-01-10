from datetime import datetime
import pandas as pd
import open_cdr as ocdr

class Cell:
    """Object representing a cell"""
    def __init__(self, id:int):
        self.weekend = []
        self.weekday = []
        self.id = id

def parse_data():
    data_list = []
    for cellid in range(1,10001):
        data_list.append(Cell(cellid))
    cdr = ocdr.merge_all()
