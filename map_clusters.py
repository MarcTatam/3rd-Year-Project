import plotly.graph_objects as go
import plotly.express as ex
import pandas as pd
import geojson
from land_use_classification import load_centroids, load_cells, attach_to_centroids

def map_clusters():
    centroids = load_centroids()
    cells = load_cells()
    centroids = attach_to_centroids(cells, centroids)
    df_list = []
    for i in range(len(centroids)):
        centroid = centroids[i]
        for cell in centroid.cells:
            print(cell.id)
            df_list.append([cell.id-1,str(i+1)])
    df = pd.DataFrame(df_list, columns=("CellID", "Centroid"))
    df.sort_values("CellID")
    print(df)
    with open("milano-grid.geojson") as f:
        gj = geojson.load(f)
    fig= ex.choropleth_mapbox(df,
                            geojson=gj,
                            locations=df.CellID,
                            color="Centroid",
                            opacity = 0.5
                           )
    fig.update_layout(mapbox_style="stamen-terrain", mapbox_center_lon=9.19, mapbox_center_lat=45.4642,mapbox_zoom=10)
    fig.update_geos(fitbounds="locations")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    #fig.show()
    fig.write_image("./fourth_attempt.png")

if __name__ == "__main__":
    map_clusters()

