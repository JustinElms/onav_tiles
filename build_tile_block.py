import os

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import box

import utils
from extract_land_geoms import extract_land_geoms
from extract_surface_contours import extract_surface_contours
from write_tiles import write_tiles

WM_BOUNDS = [-20037508.34, -20048966.1, 20037508.34, 20048966.1]


def calculate_df_bbox(row):
    x0 = row.x * 4096 - 64
    x1 = (row.x + 1) * 4096 + 64
    y0 = row.y * 4096 - 64
    y1 = (row.y + 1) * 4096 + 64
    row.ext = [x0, y0, x1, y1]
    row.geometry = box(x0, y0, x1, y1)
    return row


def calculate_block_bbox(x, y, z) -> box:
    wm_ext = utils.calculate_tile_ext(x, y, z)

    n_tiles = 2**z

    # account for lon wrapping
    if x == 0:
        wm_ext[0] = -2 * WM_BOUNDS[2] + wm_ext[0]
    if x == n_tiles - 1:
        wm_ext[2] = 2 * WM_BOUNDS[2] + wm_ext[2]

    wm_ext[:2] = wm_ext[:2] - 4096 * 3
    wm_ext[2:] = wm_ext[2:] + 4096 * 3
    wm_bbox = box(*wm_ext)

    return wm_bbox


def build_tile_block(inputs: np.array) -> gpd.GeoDataFrame:
    x, y, z, mbt_path = inputs

    print(f"Starting {[x,y,z]}")

    x = int(x)
    y = int(y)
    z = int(z)

    config = utils.read_config()

    pkl_name = f"{config["PKL_PATH"]}/contours_{x}_{y}_{z}.pkl"
    if os.path.exists(pkl_name):
        mbt_df = pd.read_pickle(pkl_name)
    else:
        wm_bbox = calculate_block_bbox(x, y, z)

        z14_tile_coords = utils.get_z14_tile_coords(x, y, z)
        land_df = extract_land_geoms(z14_tile_coords, wm_bbox)

        mbt_df = gpd.GeoDataFrame(
            columns=["x", "y", "ext", "geometry"], crs=3857
        )
        mbt_df["x"] = z14_tile_coords[:, 0]
        mbt_df["y"] = z14_tile_coords[:, 1]
        mbt_df = mbt_df.apply(calculate_df_bbox, axis=1)

        pc_ext = utils.calculate_tile_ext(x, y, z, crs="pc")

        mbt_df = extract_surface_contours(mbt_df, land_df, pc_ext)

        mbt_df.to_pickle(pkl_name)

    write_tiles(mbt_df, mbt_path)

    print(f"Completed {[x,y,z]}")
