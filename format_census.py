import fiona
import shapely
import json 
import geopandas as gpd
import pandas as pd

def format():
    out_df = pd.DataFrame(columns=["CellID","P1","E3","E4"])
    with open("milano_istat.json") as f:
        cdf = gpd.read_file(f)
    with open("milano-grid.geojson") as f:
        cell_list = json.load(f)["features"]
    for cell in cell_list:
        id = cell["id"]
        print(id)
        cell = shapely.geometry.shape(cell["geometry"])
        temp_df = cdf[cdf.overlaps(cell)].append(cdf[cdf.within(cell)]).append(cdf[cdf.covers(cell)])
        out_df.loc[len(out_df)] = [id+1, temp_df["P1"].sum(),temp_df["E3"].sum(), temp_df["E4"].sum()]
    out_df.to_csv("CensusData.csv")

def zscore():
    df = pd.read_csv("CensusData.csv", index_col=0)
    #Normalise P1
    mean = df["P1"].mean()
    sd = df["P1"].std()
    df["P1"] = (df["P1"]-mean)/sd
    #Normalise E3
    mean = df["E3"].mean()
    sd = df["E3"].std()
    df["E3"] = (df["E3"]-mean)/sd
    #Normalise E4
    mean = df["E4"].mean()
    sd = df["E4"].std()
    df["E4"] = (df["E4"]-mean)/sd
    df.to_csv("CensusDataZScore.csv")

def minmax():
    df = pd.read_csv("CensusData.csv", index_col=0)
    #Normalise P1
    minn = df["P1"].min()
    maxx = df["P1"].max()
    df["P1"] = (df["P1"]-minn)/(maxx-minn)
    #Normalise E3
    minn = df["E3"].min()
    maxx = df["E3"].max()
    df["E3"] = (df["E3"]-minn)/(maxx-minn)
    #Normalise E4
    minn = df["E4"].min()
    maxx = df["E4"].max()
    df["E4"] = (df["E4"]-minn)/(maxx-minn)
    df.to_csv("CensusDataMinMax.csv")

if __name__ == "__main__":
    minmax()
