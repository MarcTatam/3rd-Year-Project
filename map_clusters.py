import plotly.graph_objects as go
import plotly.express as ex
import pandas as pd
import geojson
import json
import keplergl
from land_use_classification import load_centroids, load_cells, attach_to_centroids, convert_to_residual

def map_clusters():
    """Generates a map of each cell and its cventroid assignment"""
    map = keplergl.KeplerGl()
    centroids = load_centroids()
    cells = load_cells()
    cells = convert_to_residual(cells)[0]
    centroids = attach_to_centroids(cells, centroids)
    df_list = []
    for i in range(len(centroids)):
        centroid = centroids[i]
        for cell in centroid.cells:
            df_list.append([cell.id-1,str(i+1)])
    df = pd.DataFrame(df_list, columns=("CellID", "Centroid"))
    df.sort_values("CellID")
    print(df)
    with open("milano-grid.geojson") as f:
        gj = json.load(f)
    chars = ["A","B","C","D","E","F","G","H","I","J","K","L",]
    for cell in gj["features"]:
        del cell["properties"]["cellId"]
        cell["properties"]["centroid"] = chars[int(df.loc[df["CellID"]==cell["id"]]["Centroid"])-1]
    map.add_data(gj,"Cells")
    #fig= ex.choropleth_mapbox(df,
    #                        geojson=gj,
    #                        locations=df.CellID,
    #                        color="Centroid",
    #                        opacity = 0.5
    #                       )
    #fig.update_layout(mapbox_style="stamen-terrain", mapbox_center_lon=9.19, mapbox_center_lat=45.4642,mapbox_zoom=10)
    #fig.update_geos(fitbounds="locations")
    #fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    #fig.show()
    #fig.write_image("./fifth_attempt.png")
    map.config = {
		'version': 'v1',
		'config': {
			'visState': {
				'filters': [],
				'layers': [{
					'id': '2ow7zdm',
					'type': 'geojson',
					'config': {
						'dataId': 'Cells',
						'label': 'Cells',
						'color': [18, 147, 154],
						'highlightColor': [252, 242, 26, 255],
						'columns': {
							'geojson': '_geojson'
						},
						'isVisible': True,
						'visConfig': {
							'opacity': 0.6,
							'strokeOpacity': 0.8,
							'thickness': 0.5,
							'strokeColor': [221, 178, 124],
							'colorRange': {
								'name': 'Global Warming',
								'type': 'sequential',
								'category': 'Uber',
								'colors': ['#5A1846',
									'#900C3F',
									'#C70039',
									'#E3611C',
									'#F1920E',
									'#FFC300'
								]
							},
							'strokeColorRange': {
								'name': 'Global Warming',
								'type': 'sequential',
								'category': 'Uber',
								'colors': ['#5A1846',
									'#900C3F',
									'#C70039',
									'#E3611C',
									'#F1920E',
									'#FFC300'
								]
							},
							'radius': 10,
							'sizeRange': [0, 10],
							'radiusRange': [0, 50],
							'heightRange': [0, 500],
							'elevationScale': 5,
							'enableElevationZoomFactor': True,
							'stroked': True,
							'filled': True,
							'enable3d': False,
							'wireframe': False
						},
						'hidden': False,
						'textLabel': [{
							'field': None,
							'color': [255, 255, 255],
							'size': 18,
							'offset': [0, 0],
							'anchor': 'start',
							'alignment': 'center'
						}]
					},
					'visualChannels': {
						'colorField': {
							'name': 'centroid',
							'type': 'string'
						},
						'colorScale': 'ordinal',
						'strokeColorField': None,
						'strokeColorScale': 'quantile',
						'sizeField': None,
						'sizeScale': 'linear',
						'heightField': None,
						'heightScale': 'linear',
						'radiusField': None,
						'radiusScale': 'linear'
					}
				}],
				'interactionConfig': {
					'tooltip': {
						'fieldsToShow': {
							'Cells': [{
								'name': 'centroid',
								'format': None
							}]
						},
						'compareMode': False,
						'compareType': 'absolute',
						'enabled': True
					},
					'brush': {
						'size': 0.5,
						'enabled': False
					},
					'geocoder': {
						'enabled': False
					},
					'coordinate': {
						'enabled': False
					}
				},
				'layerBlending': 'normal',
				'splitMaps': [],
				'animationConfig': {
					'currentTime': None,
					'speed': 1
				}
			},
			'mapState': {
				'bearing': 0,
				'dragRotate': False,
				'latitude': 45.46419671436054,
				'longitude': 9.19193271681106,
				'pitch': 0,
				'zoom': 11,
				'isSplit': False
			},
			'mapStyle': {
				'styleType': 'dark',
				'topLayerGroups': {},
				'visibleLayerGroups': {
					'label': True,
					'road': True,
					'border': False,
					'building': True,
					'water': True,
					'land': True,
					'3d building': False
				},
				'threeDBuildingColor': [9.665468314072013,
					17.18305478057247,
					31.1442867897876
				],
				'mapStyles': {}
			}
		}
	}
    map.save_to_html(file_name="landuseclassification.html")

if __name__ == "__main__":
    map_clusters()

