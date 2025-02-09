# data_loader.py - Handles data loading and spatial processing

import geemap
import ee
from pathlib import Path

def load_dw_results(selection, base_path, sub_areas):
    """
    Load Dynamic World (DW) classification results per sub-area and per year.

    Args:
        selection (dict): Dictionary containing user-selected values.
        base_path (Path): Base path for data.
        sub_areas (list): List of available sub-areas.

    Returns:
        results_per_area_and_year (dict): Dictionary storing DW classification results.
    """
    results_per_area_and_year = {}

    for sub_area in sub_areas:
        park_sub_shp = base_path / sub_area / f"{selection['Park']}_{sub_area}.shp"

        if park_sub_shp.exists():
            park_sub = geemap.shp_to_ee(str(park_sub_shp))
            geometry = park_sub.geometry()

            dw_classes_per_year = {
                Year: geemap.dynamic_world(
                    geometry, f"{Year}-01-01", f"{Year}-12-31", return_type="class", reducer="mode"
                ).clip(geometry)
                for Year in selection["Years"]
            }

            results_per_area_and_year[sub_area] = dw_classes_per_year

    return results_per_area_and_year


def load_parks_and_buffers(base_path):
    """
    Load the dissolved parks and buffers shapefile.

    Args:
        base_path (Path): Base path for data.

    Returns:
        ee.FeatureCollection: Dissolved parks feature collection.
    """
    parks_shp = base_path / "dissolved_all_buffers_FINAL.shp"
    return geemap.shp_to_ee(str(parks_shp))


def generate_fishnet(parks, base_path=None, selection=None, h_interval=1.0, v_interval=1.0):
    """
    Generate a fishnet (grid) overlaying the region of interest.

    Args:
        parks (ee.FeatureCollection): Feature collection representing dissolved parks.
        base_path (Path, optional): Base path for data.
        selection (dict, optional): User-selected park information.
        h_interval (float, optional): Horizontal grid spacing in degrees. Default is 1.0.
        v_interval (float, optional): Vertical grid spacing in degrees. Default is 1.0.

    Returns:
        ee.FeatureCollection: Generated fishnet grid.
    """
    bounding_box = parks.geometry().bounds()

    if base_path and selection:
        park_dissolved_shp = base_path / "Dissolved" / f"{selection['Park']}_Dissolved.shp"
        if park_dissolved_shp.exists():
            park_dissolved = geemap.shp_to_ee(str(park_dissolved_shp))
            bounding_box = park_dissolved.geometry().bounds()

    coords = bounding_box.getInfo()['coordinates'][0]
    min_lon, min_lat = coords[0]
    max_lon, max_lat = coords[2]

    region = ee.Geometry.BBox(min_lon, min_lat, max_lon, max_lat)
    return geemap.fishnet(region, h_interval=h_interval, v_interval=v_interval)


def process_window_geometries(selection, base_path, sub_areas, fishnet):
    """
    Process window geometries for each sub-area by intersecting with the fishnet grid.

    Args:
        selection (dict): Dictionary containing user-selected values.
        base_path (Path): Base path for data.
        sub_areas (list): List of available sub-areas.
        fishnet (ee.FeatureCollection): Fishnet grid overlaying the region.

    Returns:
        dict: Dictionary storing window geometries per sub-area.
    """
    window_geometries_per_sub_area = {}

    for sub_area in sub_areas:
        park_sub_shp = base_path / sub_area / f"{selection['Park']}_{sub_area}.shp"

        if park_sub_shp.exists():
            park_sub = geemap.shp_to_ee(str(park_sub_shp))
            geometry = park_sub.geometry()

            intersected_features = fishnet.map(
                lambda feature: ee.Feature(feature).intersection(geometry, ee.ErrorMargin(1))
            )

            try:
                feature_list = intersected_features.getInfo().get('features', [])
                window_geometries = [
                    ee.Geometry(feature['geometry']) for feature in feature_list if ee.Geometry(feature['geometry']).area().getInfo() > 1
                ]
                window_geometries_per_sub_area[sub_area] = window_geometries

            except Exception as e:
                print(f"⚠️ Warning: Failed to process geometries for {sub_area}. Error: {e}")

    return window_geometries_per_sub_area