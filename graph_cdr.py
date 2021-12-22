import plotly.graph_objects as go
import plotly.express as ex
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import open_cdr as cdr
import geojson

def graph_day_one():
    df = cdr.merge_countries("2013-11-01")
    selected_df = df.loc[df[0] == 1]
    selected_df[1] = pd.to_datetime(selected_df[1],unit='ms')
    print(selected_df)
    selected_df.plot(x=1, y=[3,4,5,6,7])
    plt.show()

def heatmap():
    df = cdr.merge_countries("2013-11-01")
    with open("milano-grid.geojson") as f:
        gj = geojson.load(f)
    points = []
    i = 0
    colours = []
    for feature in gj['features']:
        if feature['geometry']['type'] == 'Polygon':
            points.extend(feature['geometry']['coordinates'][0][::-1])    
            points.append([None, None]) # mark the end of a polygon
            if i % 2 == 0:
                colours.append("red")
            else:
                colours.append("blue")
    lons, lats = zip(*points)
    fig = go.Figure(go.Scattermapbox(mode = "lines", fill = "toself",lon =lons,lat =lats, fillcolor =colours))
    fig.update_layout(mapbox_style="stamen-terrain", mapbox_center_lon=180)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()

def heatmap2():
    df = cdr.merge_countries("2013-11-01")
    df = df[df[1] == 1383314400000]
    df.columns =['CellID', 'Timestamp', 'SMSIN','SMSOUT','CALLIN','CALLOUT','INTERNET']
    df['SMSIN'] = np.log2(df['SMSIN'])
    print(df[df['CellID']==5161])
    with open("milano-grid.geojson") as f:
        gj = geojson.load(f)
    fig= go.Figure(go.Choroplethmapbox(
                            geojson=gj,
                            locations=df.CellID,
                            z=df.SMSIN,
                            colorscale="jet",  
                            marker_line_width=0,

                            marker=dict(opacity=0.5)
                           ))
    fig.update_layout(mapbox_style="stamen-terrain", mapbox_center_lon=180)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()

if __name__ == "__main__":
    heatmap2()

