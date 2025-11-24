import gzip
import sqlite3

import geojson
import geopandas as gpd
import mapbox_vector_tile
import numpy as np
import pandas as pd
from shapely import (
    box,
    empty,
    GeometryType,
    make_valid,
    transform,
)

import utils


def wm_to_global_tile_coords(coords: np.array) -> np.array:

    wm_proj_bounds = [-20037508.34, -20048966.1, 20037508.34, 20048966.1]

    x_vals = coords[:, 0]
    y_vals = coords[:, 1]

    z14_x_coords = (
        4096
        * (2**14)
        * (x_vals - wm_proj_bounds[0])
        / (wm_proj_bounds[2] - wm_proj_bounds[0])
    )
    z14_y_coords = (
        4096
        * (2**14)
        * (y_vals - wm_proj_bounds[1])
        / (wm_proj_bounds[3] - wm_proj_bounds[1])
    )

    tile_based_coords = np.vstack([z14_x_coords, z14_y_coords]).T
    tile_based_coords = np.round(tile_based_coords).astype(int)

    return np.array(tile_based_coords, dtype=np.float64)


def read_osm_land_polys(wm_bbox: np.array) -> gpd.GeoDataFrame:

    config = utils.read_config()

    land_df = gpd.read_file(
        config["OSM_LAND_POLY_PATH"],
        bbox=tuple(wm_bbox.bounds),
    )
    land_df = land_df.set_geometry(land_df.geometry.intersection(wm_bbox))
    land_df = land_df[~land_df.geometry.is_empty]
    land_df.geometry = land_df.transform(wm_to_global_tile_coords)
    land_df.drop(columns=["FID"], inplace=True)
    land_df["level"] = 0
    land_df["type"] = "land"
    land_df = land_df.dissolve(by=["level", "type"], as_index=False)

    return land_df


def global_tile_coords(
    coords: np.array,
    x: int,
    y: int,
) -> np.array:

    x_vals = coords[:, 0] + x * 4096
    y_vals = coords[:, 1] + y * 4096

    return np.array([x_vals, y_vals], dtype=np.float64).T


def globalize_df_geoms(row):
    new_geom = empty(1, geom_type=GeometryType.POLYGON)[0]
    if row.geometry:
        new_geom = transform(
            row.geometry, lambda xy: global_tile_coords(xy, row.x, row.y)
        )
    row.geometry = new_geom

    return row


def extract_water_geoms(mbt_path, x_bounds, y_bounds, z) -> gpd.GeoDataFrame:
    water_df = gpd.GeoDataFrame(
        columns=["layer", "id", "class", "x", "y", "z", "geometry"],
        geometry="geometry",
        crs="EPSG:3857",
    )
    mbt_data = utils.extract_mbt(mbt_path, x_bounds, y_bounds, z)
    mbt_gdf = gpd.read_file(mbt_data)

    if len(mbt_gdf) > 0:

        mbt_gdf = mbt_gdf[water_df.columns]

        mbt_gdf.set_crs(3857, inplace=True, allow_override=True)

        if "class" in mbt_gdf.columns:
            water_df = pd.concat(
                [
                    water_df,
                    mbt_gdf[
                        (mbt_gdf["class"] == "river") | (mbt_gdf["class"] == "lake")
                    ],
                ],
                ignore_index=True,
            )
            invalid_idxs = water_df.loc[~water_df.geometry.is_valid].index
            for idx in invalid_idxs:
                water_df.loc[idx, "geometry"] = make_valid(
                    water_df.loc[idx, "geometry"]
                )
    water_df = water_df.explode()
    water_df = water_df.loc[water_df.geometry.geometry.type == "Polygon"]
    water_df.drop(columns=["layer", "id", "class"], inplace=True)
    water_df = water_df.apply(globalize_df_geoms, axis=1)

    return water_df


def extract_land_geoms(xy_coords: np.array, wm_bbox: box) -> gpd.GeoDataFrame:

    config = utils.read_config()

    x_bounds = [np.min(xy_coords[:, 0]), np.max(xy_coords[:, 0])]
    y_bounds = [np.min(xy_coords[:, 0]), np.max(xy_coords[:, 0])]

    land_df = read_osm_land_polys(wm_bbox)
    water_df = extract_water_geoms(config["OSM_MBT_PATH"], x_bounds, y_bounds, 14)

    land_df.geometry = land_df.difference(water_df.union_all())

    return land_df
