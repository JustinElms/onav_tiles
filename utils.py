import configparser

import cartopy.crs as ccrs
import numpy as np


def tile_coords_to_ll(coords: np.array, x: int, y: int, z: int) -> np.array:
    n_tiles = 2**z
    x_vals = coords[:, 0]
    y_vals = coords[:, 1]

    xtile = x + (x_vals / 4096)
    ytile = n_tiles - (y + 1) + ((4096 - y_vals) / 4096)
    lon_deg = (xtile / n_tiles) * 360.0 - 180.0
    lat_rad = np.atan(np.sinh(np.pi * (1 - 2 * ytile / n_tiles)))
    lat_deg = np.rad2deg(lat_rad)
    return np.vstack((lon_deg, lat_deg)).T


def calculate_tile_ext(x: int, y: int, zoom: int, crs: str = "wm") -> np.array:
    pc_crs = ccrs.PlateCarree()
    wm_crs = ccrs.epsg("3857")  # web mercator projection
    tile_coord_bounds = np.array([-64, -64, 4160, 4160])

    pc_ext = tile_coords_to_ll(
        np.array([tile_coord_bounds[[0, 1]], tile_coord_bounds[[2, 3]]]), x, y, zoom
    ).flatten()
    if crs == "pc":
        return pc_ext

    wm_ext = wm_crs.transform_points(pc_crs, pc_ext[[0, 2]], pc_ext[[1, 3]])
    return np.array([wm_ext[0, 0], wm_ext[0, 1], wm_ext[1, 0], wm_ext[1, 1]])


def get_z14_tile_coords(x: int, y: int, z: int) -> np.array:
    """
    returns indicies of zoom level 14 tiles that span tile specified by input coordinates
    """

    x14_0 = 2**14 * x / 2**z
    x14_1 = (2**14 * (x + 1) / 2**z) - 1
    y14_0 = 2**14 * y / 2**z
    y14_1 = (2**14 * (y + 1) / 2**z) - 1

    x14_idxs = np.arange(x14_0, x14_1 + 1)
    y14_idxs = np.arange(y14_0, y14_1 + 1)

    z14_xy = np.array(np.meshgrid(x14_idxs, y14_idxs)).T.reshape(-1, 2).astype(int)

    return z14_xy


def get_tile_indexes(zoom) -> list:
    xyz = np.array(
        np.meshgrid(range(2**zoom), range(2**zoom), [zoom], indexing="ij")
    ).T.reshape(-1, 3)

    xyz[:, 1] = xyz[::-1, 1]

    return xyz.astype(int)


def read_config() -> dict:
    config = configparser.ConfigParser()

    config.read("onav_tiles.cfg")

    return config["DEFAULT"]
