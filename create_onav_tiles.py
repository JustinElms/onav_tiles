import glob
import gzip
import io
import os
import random
import sqlite3
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import product, repeat
from pathlib import Path

import cartopy.crs as ccrs
import cv2
import geopandas as gpd
import mapbox_vector_tile
import numpy as np
import pandas as pd
import pyogrio
import rasterio
import xarray as xr
from osgeo import gdal
from rasterio.features import rasterize, shapes
from PIL import Image
from skimage.morphology import remove_small_objects
from shapely import (
    box,
    Polygon,
    MultiPolygon,
)
from shapely.affinity import scale
from shapely.geometry import shape

MAX_WORKERS = 4

ONAV_BASEMAP_DB = "/data/misc/shapes/onav_basemap.mbtiles"
ONAV_LAND_MASK_DB = "/data/misc/shapes/onav_land_mask.mbtiles"
ONAV_BATHY_DB = "/data/misc/shapes/onav_bathy.mbtiles"
ETOPO_PATH = "/data/misc/ETOPO2022/"
TILE_IMG_PATH = "/data/misc/shapes/tile_images/"


MBT_PATH = "/data/misc/maptiler-osm-2020-02-10-v3.11-planet.mbtiles"

TILE_SIZE = 256
TILE_BUFFER = 4

MAX_EXTENT = 20037508.342789244
WORLD_SIZE = MAX_EXTENT * 2

TILE_COLORS = {
    "land": {
        0: "f6e8c3",
        100: "f0deb0",
        200: "e8d29a",
        500: "e1c684",
        1000: "d8b46c",
        2000: "cd9d50",
        3000: "c28633",
        4000: "af7222",
        5000: "9b5f14",
        6000: "844c09",
        8000: "6c3e07",
        15000: "543005",
    },
    "bathy": {
        -0: "8a8989",
        -100: "7e7d7d",
        -200: "737372",
        -500: "6a6a69",
        -1000: "626161",
        -2000: "5a5a5a",
        -3000: "535352",
        -4000: "4c4b4b",
        -5000: "474646",
        -6000: "414140",
        -8000: "3c3b3b",
        -15000: "373737",
    },
    "ice": {
        0: "8683bd",
        100: "8d89c0",
        200: "9390c3",
        500: "9a96c6",
        1000: "a19eca",
        2000: "aaa8d0",
        3000: "b2b2d5",
        4000: "bcbddc",
        5000: "c5c6e1",
        6000: "cfd0e6",
        8000: "dadaeb",
        15000: "e2e2ef",
    },
}

pc_crs = ccrs.PlateCarree()
wm_crs = ccrs.epsg("3857")

contour_levels = np.array(
    [100, 200, 500, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 15000]
)
contour_levels = np.array([*contour_levels[::-1] * -1, *contour_levels])


def get_etopo_idx(etopo_file: str) -> str:
    etopo_idx = etopo_file.split("ETOPO_2022_v1_15s_")[-1].split("_surface.nc")[0]

    return etopo_idx


def etopo_grid_coords() -> gpd.GeoDataFrame:
    etopo_files = sorted(
        glob.glob("/data/misc/ETOPO2022/15s/15s_surface_elev_netcdf/*.nc")
    )

    idx_coords = []
    for etopo_file in etopo_files:
        etopo_idx = get_etopo_idx(str(etopo_file))
        y_dir = etopo_idx[0]
        x_dir = etopo_idx[3]
        y = int(etopo_idx[1:3])
        x = int(etopo_idx[4:])

        if y_dir == "N":
            lat_max = int(y)
            y = y + 90
        else:
            lat_max = -int(y)
            y = 90 - y

        lat_min = lat_max - 15

        y_idx = int((180 - y) / 15)

        if x_dir == "W":
            lon_min = -int(x)
            x = x + 180
        else:
            lon_min = int(x)
            x = 180 - x

        lon_max = lon_min + 15

        x_idx = int((360 - x) / 15)

        wm_bound_coords = wm_crs.transform_points(
            pc_crs, np.array([lon_min, lon_max]), np.array([lat_min, lat_max])
        )

        wm_bbox = box(
            wm_bound_coords[0, 0],
            wm_bound_coords[0, 1],
            wm_bound_coords[1, 0],
            wm_bound_coords[1, 1],
        )

        idx_coords.append([etopo_file, etopo_idx, x_idx, y_idx, wm_bbox])

    return gpd.GeoDataFrame(
        idx_coords,
        columns=["etopo_file", "etopo_idx", "x", "y", "geometry"],
    )


def query_mbt(x, y, z) -> gpd.GeoDataFrame:
    query = """
        SELECT tile_data 
        FROM tiles 
        WHERE zoom_level = ? AND tile_column = ? AND tile_row = ?
    """

    y = 2**z - y - 1

    conn = sqlite3.connect(MBT_PATH)
    cursor = conn.cursor()
    cursor.execute(query, (z, x, y))
    row = cursor.fetchone()

    tile_bytes = row[0]

    decompressed_data = gzip.decompress(tile_bytes)
    decoded_tile = mapbox_vector_tile.decode(decompressed_data)

    ice_df = None
    ice_features = decoded_tile.get("landcover")
    if ice_features is not None:
        ice_df = gpd.GeoDataFrame.from_features(ice_features["features"])
        ice_df = ice_df.loc[ice_df["class"] == "ice"]
        ice_df["class"] = "ice"

    water_df = None
    water_features = decoded_tile.get("water")
    if water_features is not None:
        water_df = gpd.GeoDataFrame.from_features(water_features["features"])
        water_df["class"] = "water"

    if ice_df is None and water_df is None:
        tile_df = gpd.GeoDataFrame({"geometry": [], "class": []}, crs=3857)
    else:
        tile_df = pd.concat([water_df, ice_df]).explode()

    return tile_df


def extract_mbt_data(x, y, z) -> gpd.GeoDataFrame:
    x_max = 2**z - 1

    tile_df = query_mbt(x, y, z)

    if x == 0:
        left_tile = query_mbt(x_max, y, z)
        left_tile.geometry = left_tile.translate(xoff=-4096)

        tile_df = pd.concat([tile_df, left_tile])

    if x == x_max:
        right_tile = query_mbt(0, y, z)
        right_tile.geometry = right_tile.translate(xoff=4096)

        tile_df = pd.concat([tile_df, right_tile])

    tile_df.geometry = tile_df.scale(
        xfact=TILE_SIZE / 4096, yfact=TILE_SIZE / 4096, origin=(0, 0)
    )

    tile_df = tile_df[["geometry", "class"]].dissolve(by="class").reset_index()

    tile_df.geometry = (
        tile_df.clip(
            [
                -TILE_BUFFER,
                -TILE_BUFFER,
                TILE_SIZE + TILE_BUFFER,
                TILE_SIZE + TILE_BUFFER,
            ],
            sort=True,
        )
        .translate(xoff=TILE_BUFFER, yoff=TILE_BUFFER)
        .make_valid(method="structure", keep_collapsed=False)
    )

    return tile_df


def combine_adjacent_by_buffer(geometry_series, dist):
    """Buffers, merges, and shrinks a group of geometries."""
    buffered_out = geometry_series.buffer(dist)

    unified = buffered_out.union_all()

    return unified.buffer(-dist)


def wm_to_tile_coords(wm_coords, zoom) -> np.ndarray:
    wm_x = wm_coords[:, 0]
    wm_y = wm_coords[:, 1]

    n_tiles = 2**zoom

    px_x = TILE_SIZE * n_tiles * (wm_x + MAX_EXTENT) / (2 * MAX_EXTENT)
    px_y = TILE_SIZE * n_tiles * (MAX_EXTENT - wm_y) / (2 * MAX_EXTENT)

    px_coords = np.vstack([px_x.astype(int), px_y.astype(int)]).T

    return np.array(px_coords, dtype=np.float64)


def find_contours(surface: np.array, levels: np.array) -> np.array:
    contours = surface.copy()
    contours[~np.isnan(contours)] = np.min(levels)

    for level in levels[:-1]:
        contours[surface >= level] = level

    return contours


def cv2coords(contour: np.array, x_coords: np.array, y_coords: np.array) -> Polygon:
    coords = np.vstack(
        [
            x_coords[
                contour[:, 0, 1].astype(int),
                contour[:, 0, 0].astype(int),
            ],
            y_coords[
                contour[:, 0, 1].astype(int),
                contour[:, 0, 0].astype(int),
            ],
        ]
    ).T
    return coords


def get_contour_polys(contours, x_coords, y_coords, levels):
    contour_df = gpd.GeoDataFrame(
        columns=["geometry", "level", "layer_type"],
        geometry="geometry",
        crs=3857,
    )
    for level in levels:
        mask = contours.copy()
        mask[mask != level] = np.nan
        mask[mask == level] = 1
        mask[mask != 1] = 0
        mask[:, 0] = 0
        mask[:, -1] = 0
        mask[0, :] = 0
        mask[-1, :] = 0

        if np.nansum(mask) == 0:
            continue

        level_contours, hierarchy = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )

        hierarchy = hierarchy[0]
        level_polys = []
        for idx, h in enumerate(hierarchy):
            p_ext, p_previous, p_child, p_parent = h
            if p_parent == -1:
                ext = level_contours[idx]
                holes = []
                if len(ext) > 4:
                    ext = cv2coords(ext, x_coords, y_coords)
                    child_idxs = np.argwhere(hierarchy[:, 3] == idx).flatten()
                    if len(child_idxs) > 0:
                        holes = [
                            cv2coords(level_contours[c], x_coords, y_coords)
                            for c in child_idxs
                        ]

                    level_polys.append(Polygon(shell=ext, holes=holes))

        if len(level_polys) > 0:
            level_df = gpd.GeoDataFrame(
                {
                    "geometry": [MultiPolygon(level_polys)],
                    "level": [level],
                },
                crs=3857,
            )
            level_df.geometry = level_df.make_valid(
                method="structure", keep_collapsed=False
            )
            contour_df = pd.concat([contour_df, level_df], ignore_index=True)
    return contour_df


def extract_etopo_contours(etopo_file: str | Path) -> None:
    if not isinstance(etopo_file, Path):
        etopo_file = Path(etopo_file)

    etopo_idx = etopo_file.name.split("ETOPO_2022_v1_15s_")[-1].split("_surface.nc")[0]

    contour_path = f"/data/misc/ETOPO_contours/contour_{etopo_idx}.pkl"
    if os.path.exists(contour_path):
        # print(f"Contours from {etopo_file} previously processed.")
        contour_df = pd.read_pickle(contour_path)
    else:
        # print(f"Extracting contours from {etopo_file}")
        etopo_ds = xr.open_dataset(etopo_file).sel(lat=slice(-86, 86))

        lat = etopo_ds.lat.data
        lon = etopo_ds.lon.data

        lon_grid, lat_grid = np.meshgrid(lon, lat)
        wm_coords = wm_crs.transform_points(pc_crs, lon_grid, lat_grid)[:, :, :2]
        wm_x_coords = wm_coords[:, :, 0]
        wm_y_coords = wm_coords[:, :, 1]

        contour_mask = find_contours(etopo_ds.z.data, contour_levels)
        contour_df = get_contour_polys(
            contour_mask, wm_x_coords, wm_y_coords, contour_levels
        )

        contour_df.geometry = contour_df.make_valid(
            method="structure", keep_collapsed=False
        )

        contour_df.to_pickle(contour_path)
        # print(f"Saved contours to {contour_path}")

    return contour_df


def etopo_px_contours(etopo_file: str | Path, zoom: int) -> None:
    if not isinstance(etopo_file, Path):
        etopo_file = Path(etopo_file)

    etopo_idx = etopo_file.name.split("ETOPO_2022_v1_15s_")[-1].split("_surface.nc")[0]

    px_contour_path = Path(
        f"/data/misc/ETOPO_contours/z{zoom}/contour_px_{etopo_idx}_EPSE3857_z{zoom}.pkl"
    )

    if not px_contour_path.is_file():
        # print(f"Processing {px_contour_path}.")
        px_contour_path.parent.mkdir(parents=True, exist_ok=True)

        etopo_df = extract_etopo_contours(etopo_file)

        etopo_df.geometry = etopo_df.transform(
            lambda xy: wm_to_tile_coords(xy, zoom)
        ).make_valid(method="structure", keep_collapsed=False)

        etopo_df.to_pickle(px_contour_path)

    else:
        etopo_df = pd.read_pickle(px_contour_path)
        # print(f"{px_contour_path} previously completed.")

    return etopo_df


def get_topo_contours(x, y, z) -> gpd.GeoDataFrame:
    tile_px_bounds = np.array(
        [
            x * TILE_SIZE - TILE_BUFFER,
            y * TILE_SIZE - TILE_BUFFER,
            (x + 1) * TILE_SIZE + TILE_BUFFER,
            (y + 1) * TILE_SIZE + TILE_BUFFER,
        ]
    )

    etopo_coords = etopo_grid_coords()
    etopo_coords.geometry = etopo_coords.transform(lambda xy: wm_to_tile_coords(xy, z))

    tile_topo_df = etopo_coords[
        etopo_coords.geometry.intersects(
            box(
                (x - 1) * TILE_SIZE,
                (y - 1) * TILE_SIZE,
                (x + 2) * TILE_SIZE,
                (y + 2) * TILE_SIZE,
            )
        )
    ]

    tile_topo_df["xoff"] = None

    if tile_topo_df.x.min() == 0:
        wrap_df = etopo_coords.loc[
            (etopo_coords.x == 23)
            & (etopo_coords.y >= tile_topo_df.y.min())
            & (etopo_coords.y <= tile_topo_df.y.max())
        ]
        wrap_df["xoff"] = -1 * (2**z) * TILE_SIZE

        tile_topo_df = pd.concat([tile_topo_df, wrap_df])

    elif tile_topo_df.x.max() == 23:
        wrap_df = etopo_coords.loc[
            (etopo_coords.x == 0)
            & (etopo_coords.y >= tile_topo_df.y.min())
            & (etopo_coords.y <= tile_topo_df.y.max())
        ]
        wrap_df["xoff"] = (2**z) * TILE_SIZE
        tile_topo_df = pd.concat([tile_topo_df, wrap_df])

    etopo_dfs = []
    for _, row in tile_topo_df.iterrows():
        etopo_df = etopo_px_contours(row.etopo_file, z)

        if row.xoff is not None:
            etopo_df.geometry = etopo_df.translate(xoff=row.xoff).make_valid(
                method="structure", keep_collapsed=False
            )

        if len(etopo_df) > 0:
            etopo_dfs.append(etopo_df)

    combined_df = pd.concat(etopo_dfs)

    grouped_geometry = combined_df.groupby("level")["geometry"].apply(
        lambda x: combine_adjacent_by_buffer(x, 0.75)
    )

    contour_px_df = gpd.GeoDataFrame(
        grouped_geometry, crs=combined_df.crs
    ).reset_index()

    tile_contours = contour_px_df.explode().reset_index()

    tile_contours = tile_contours[tile_contours.area >= 4]

    tile_contours = tile_contours.clip(tile_px_bounds).sort_index()
    tile_contours.geometry = tile_contours.make_valid(
        method="structure", keep_collapsed=False
    )

    tile_contours.geometry = tile_contours.translate(
        xoff=-tile_px_bounds[0], yoff=-tile_px_bounds[1]
    )

    return tile_contours


def create_tile_mask(x, y, z) -> np.ndarray:

    mbt_df = extract_mbt_data(x, y, z)

    tile_shape = (TILE_SIZE + 2 * TILE_BUFFER, TILE_SIZE + 2 * TILE_BUFFER)

    ice_mask = None
    if "ice" in mbt_df["class"].values:
        ice_mask = rasterize(
            mbt_df.loc[mbt_df["class"] == "ice"].geometry,
            out_shape=tile_shape,
            all_touched=True,
        )

    water_mask = None
    if "water" in mbt_df["class"].values:
        water_mask = rasterize(
            mbt_df.loc[mbt_df["class"] == "water"].geometry,
            out_shape=tile_shape,
            all_touched=True,
        )

    tile_mask = np.ones(tile_shape)

    if water_mask is not None:
        tile_mask[water_mask == 1] = 0
    if ice_mask is not None:
        tile_mask[ice_mask == 1] = 2

    tile_bounds = mbt_df.total_bounds
    if y == 0:
        tile_mask[int(tile_bounds[3]) :, :] = 0
    elif y == 2**z - 1:
        tile_mask[: int(tile_bounds[1]), :] = 2

    return np.flipud(tile_mask)


def create_tile(x, y, z) -> None:
    x = int(x)
    y = int(y)
    z = int(z)

    print(f"Starting tile ({x}, {y}, {z}).")

    tile_contours = get_topo_contours(x, y, z)
    tile_mask = create_tile_mask(x, y, z)

    tile_img = np.zeros(
        (TILE_SIZE + 2 * TILE_BUFFER, TILE_SIZE + 2 * TILE_BUFFER, 3), dtype=int
    )

    for layer_type, mask_value in zip(["bathy", "land", "ice"], [0, 1, 2]):
        if ~np.any(tile_mask == mask_value):
            continue

        for _, row in tile_contours.iterrows():
            if row.geometry.geom_type == "MultiPolygon":
                row_geom = row.geometry.geoms
            else:
                row_geom = [row.geometry]

            contour_mask = rasterize(
                row_geom,
                out_shape=(TILE_SIZE + 2 * TILE_BUFFER, TILE_SIZE + 2 * TILE_BUFFER),
                all_touched=True,
            )

            level_color = TILE_COLORS[layer_type].get(row.level)
            if level_color is None:
                level_color = TILE_COLORS[layer_type][0]

            r, g, b = bytes.fromhex(level_color)
            tile_img[(tile_mask == mask_value) & (contour_mask == 1)] = [r, g, b]

    tile_img = tile_img.astype(np.uint8)
    tile_img = cv2.inpaint(
        tile_img,
        (255 * np.all(tile_img == 0, axis=-1)).astype(np.uint8),
        3,
        cv2.INPAINT_TELEA,
    )

    water_mask_shapes = shapes(
        tile_mask,
        mask=(tile_mask == 0),
        transform=rasterio.Affine.identity(),
    )

    water_mask_polygons = []
    for geom, value in water_mask_shapes:
        water_mask_polygons.append(shape(geom))

    water_mask_contours = gpd.GeoDataFrame(
        geometry=water_mask_polygons, crs=tile_contours.crs
    )

    bathy_contours = gpd.overlay(
        tile_contours.reset_index(drop=True),
        water_mask_contours.reset_index(drop=True),
        how="intersection",
    )

    all_boundaries = bathy_contours.geometry.boundary.union_all()
    outer_perimeter = box(
        0, 0, TILE_SIZE + 2 * TILE_BUFFER, TILE_SIZE + 2 * TILE_BUFFER
    ).boundary
    internal_borders = all_boundaries.difference(outer_perimeter)
    cleaned_bathy_contours = gpd.GeoDataFrame(
        geometry=[internal_borders], crs=bathy_contours.crs
    )

    bathy_contours.geometry = bathy_contours.translate(
        xoff=-TILE_BUFFER, yoff=-TILE_BUFFER
    )
    bathy_contours.geometry = bathy_contours.geometry.apply(
        lambda geom: scale(
            geom,
            xfact=4096 / TILE_SIZE,
            yfact=-4096 / TILE_SIZE,
            origin=(0, 0),
        )
    )
    bathy_contours.geometry = bathy_contours.translate(yoff=4096)

    if len(bathy_contours) > 0:
        write_vector_tile(
            y,
            x,
            z,
            bathy_contours.geometry.explode(),
            ONAV_BATHY_DB,
        )

    tile_img = tile_img[TILE_BUFFER:-TILE_BUFFER, TILE_BUFFER:-TILE_BUFFER, :]
    land_mask = tile_mask.copy()
    land_mask[land_mask == 2] = 1
    land_mask = land_mask[TILE_BUFFER:-TILE_BUFFER, TILE_BUFFER:-TILE_BUFFER]

    write_tile_blob(y, x, z, land_mask.astype(np.uint8), ONAV_LAND_MASK_DB)

    image_path = f"{TILE_IMG_PATH}onav_{x}_{y}_{z}.png"

    img = Image.fromarray(tile_img)
    img.save(image_path)

    with open(image_path, "rb") as file:
        image_blob = file.read()
        write_tile_blob(y, x, z, image_blob, ONAV_BASEMAP_DB)

    print(f"Completed tiles ({x}, {y}, {z}).")


def write_tile_blob(x: int, y: int, z: int, blob_data: bytes, mbt_path: str) -> None:
    con = sqlite3.connect(mbt_path, timeout=30.0, isolation_level=None)
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA busy_timeout=5000;")

    attempt = 0
    max_retries = 10
    base_delay = 0.1
    success = False
    while attempt < max_retries and not success:
        try:
            cur.execute("BEGIN IMMEDIATE;")
            cur.execute(
                "INSERT INTO tiles VALUES (?, ?, ?, ?)",
                (z, int(y), int(x), blob_data),
            )
            con.commit()
            success = True
        except Exception as e:
            if "database is locked" in str(e):
                attempt += 1
                con.rollback()

                max_delay = base_delay * 2**attempt
                sleep_time = random.uniform(0, max_delay)
                time.sleep(sleep_time)
            else:
                con.rollback()
                # print(e)
                # print(f"Unable to save tile ({x}, {y}, {z}) to {mbt_path}")
                return
        finally:
            if attempt == max_retries or success:
                con.close()


def write_vector_tile(
    x: int, y: int, z: int, tile_contours: gpd.GeoSeries, mbt_path: str
) -> None:
    features = [
        {
            "geometry": contour,
            "properties": {
                "layer": "bathymetry",
            },
        }
        for contour in tile_contours
    ]
    tile_data = gzip.compress(
        mapbox_vector_tile.encode(
            {
                "name": "bathymetry",
                "features": features,
            }
        )
    )

    con = sqlite3.connect(mbt_path, timeout=30.0, isolation_level=None)
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA busy_timeout=5000;")

    attempt = 0
    max_retries = 5
    base_delay = 0.1
    success = False
    while attempt < max_retries and not success:
        try:
            cur.execute("BEGIN IMMEDIATE;")
            cur.execute(
                "INSERT INTO tiles VALUES (?, ?, ?, ?)",
                (z, int(y), int(x), tile_data),
            )
            con.commit()
            success = True
        except Exception as e:
            if "database is locked" in str(e):
                attempt += 1
                con.rollback()

                max_delay = base_delay * 2**attempt
                sleep_time = random.uniform(0, max_delay)
                time.sleep(sleep_time)
            else:
                con.rollback()
                # print(e)
                # print(f"Unable to save tile ({x}, {y}, {z}) to {mbt_path}")
                return
        finally:
            if attempt == max_retries or success:
                con.close()


def adapt_array(arr):
    """Converts a NumPy array into a binary format for SQLite."""
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(blob):
    """Converts a binary BLOB back into a NumPy array."""
    out = io.BytesIO(blob)
    out.seek(0)
    return np.load(out)


def create_mbt_db(name: str) -> None:

    con = sqlite3.connect(name)
    cur = con.cursor()

    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")

    init_statements = [
        "CREATE TABLE metadata (name text, value text);",
        "CREATE TABLE tiles (zoom_level integer, tile_column integer, tile_row integer, tile_data blob);",
        "CREATE UNIQUE INDEX name on metadata (name);",
        "CREATE UNIQUE INDEX tile_index on tiles (zoom_level, tile_column, tile_row);",
    ]
    for stmt in init_statements:
        cur.execute(stmt)
    con.close()


def process_tiles(z) -> None:
    n_tiles = 2**z

    tile_x, tile_y = np.meshgrid(range(n_tiles), range(n_tiles))

    tile_idxs = np.stack(
        [tile_x.ravel(), tile_y.ravel(), np.repeat(z, n_tiles**2)]
    ).T.astype(np.int_)

    print(tile_idxs.shape, (2**z) ** 2)
    tile_groups = np.array_split(tile_idxs, MAX_WORKERS)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(process_tile_group, tile_group)
            for tile_group in tile_groups
        ]
        for future in futures:
            if future.exception():
                print(f"A task crashed with error: {future.exception()}")

    # for tile_group in tile_groups:
    #     process_tile_group(tile_group)


def process_tile_group(tile_group):

    filtered_group = filter_tile_group(tile_group)
    # for x, y, z in filtered_group:
    #     create_tile(x, y, z)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(create_tile, x, y, z) for x, y, z in filtered_group]
        for future in futures:
            if future.exception():
                print(f"A task crashed with error: {future.exception()}")


def filter_tile_group(tile_group):
    x = tile_group[:, 0]
    y = tile_group[:, 1]
    z = tile_group[:, 2]

    x_min = int(x.min())
    x_max = int(x.max())
    y_min = int(y.min())
    y_max = int(y.max())

    zoom = z[0]

    try:
        connection = sqlite3.connect(ONAV_BASEMAP_DB)
        selector = connection.cursor()
        sqlite = (
            f"SELECT tile_column, tile_row FROM tiles WHERE zoom_level = {zoom} "
            f"AND tile_column >= {x_min} "
            f"AND tile_column <= {x_max} "
            f"AND tile_row >= {y_min} "
            f"AND tile_row <= {y_max}"
        )
        selector.execute(sqlite)
        tile_coords = np.array(selector.fetchall())

        mask = (tile_group[:, None, :2] == tile_coords).all(axis=2).any(axis=1)

        filtered = tile_group[~mask]

    except Exception as e:
        print(e)
        filtered = tile_group

    return filtered


sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)

if __name__ == "__main__":
    gdal.SetConfigOption("OGR_INTERLEAVED_READING", "YES")

    if not os.path.exists(ONAV_BASEMAP_DB):
        create_mbt_db(ONAV_BASEMAP_DB)

    if not os.path.exists(ONAV_LAND_MASK_DB):
        create_mbt_db(ONAV_LAND_MASK_DB)

    if not os.path.exists(ONAV_BATHY_DB):
        create_mbt_db(ONAV_BATHY_DB)

    etopo_files = sorted(
        glob.glob("/data/misc/ETOPO2022/15s/15s_surface_elev_netcdf/*.nc")
    )

    # with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
    #     executor.map(extract_etopo_contours, etopo_files)

    # for etopo_file in etopo_files:
    #     with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
    #         executor.map(etopo_px_contours, repeat(etopo_file), range(1, 15))

    for z in range(5, 11):
        process_tiles(z)
    # process_tiles()

    # for x in range(2**5):
    #     for y in range(2**5):
    #         create_tile(x, y, 5)
