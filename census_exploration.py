import fiona
import shapely
import keplergl
import json
import sys
sys.path.append(r"C:\Users\Marc\source\repos\3rd-Year-Project\project env\Lib\site-packages")
import geopandas


def open_census_isat():
    with open("milano_istat.json","r") as f:
        struct = json.load(f)
    this_map = keplergl.KeplerGl()
    this_map.add_data(struct)
    this_map.config = {
	'version': 'v1',
	'config': {
		'mapState': {
			'bearing': 0,
			'dragRotate': False,
			'latitude': 45.455934682435,
			'longitude': 9.23461965432608,
			'pitch': 0,
			'zoom': 10.451941083083048,
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
    this_map.save_to_html(data={"data_1":struct}, 
	file_name='Census 1.html',config=this_map.config)



if __name__ == "__main__":
	with open("milano_istat.json","r") as f:
		df = geopandas.read_file(f)
	print(df.columns)
	print(df[df.contains(shapely.geometry.Point(9.123932, 45.478733))])
