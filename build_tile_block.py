import os

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import box, transform

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


def global_to_tile_coords(coords, x, y) -> np.array:
    x_vals = coords[:, 0]
    y_vals = coords[:, 1]

    x_min = x * 4096
    y_min = y * 4096

    x_coords = x_vals - x_min
    y_coords = y_vals - y_min

    tile_based_coords = np.vstack([x_coords, y_coords]).T

    return np.array(tile_based_coords, dtype=np.float64)


def global_tile_transform(row):
    row.geometry = transform(
        row.geometry, lambda xy: global_to_tile_coords(xy, row.x, row.y)
    )

    return row


def build_tile_block(inputs: np.array) -> gpd.GeoDataFrame:
    x, y, z, mbt_path = inputs

    idx_str = ", ".join(map(str, [x, y, z]))
    print(f"Starting ({idx_str})")

    x = int(x)
    y = int(y)
    z = int(z)

    config = utils.read_config()

    try:
        pkl_name = f"{config["PKL_PATH"]}/contours_{x}_{y}_{z}.pkl"
        if os.path.exists(pkl_name):
            mbt_df = pd.read_pickle(pkl_name)
        else:
            wm_bbox = calculate_block_bbox(x, y, z)

            z14_tile_coords = utils.get_z14_tile_coords(x, y, z)
            land_df = extract_land_geoms(z14_tile_coords, wm_bbox)

            mbt_df = gpd.GeoDataFrame(
                columns=["x", "y", "z", "ext", "geometry"], crs=3857
            )
            mbt_df["x"] = z14_tile_coords[:, 0]
            mbt_df["y"] = z14_tile_coords[:, 1]
            mbt_df["z"] = 14
            mbt_df = mbt_df.apply(calculate_df_bbox, axis=1)

            pc_ext = utils.calculate_tile_ext(x, y, z, crs="pc")

            mbt_df = extract_surface_contours(mbt_df, land_df, pc_ext)

            mbt_df.to_pickle(pkl_name)

        mbt_df = mbt_df.apply(global_tile_transform, axis=1)
        write_tiles(mbt_df, mbt_path)

        print(f"Completed ({idx_str})")
    except Exception as e:
        with open(config["LOG_FILE"], "a") as f:
            f.write(f"({idx_str})\n")
            print(f"Error in tile ({idx_str})")
            print(e)
