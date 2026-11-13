import gc
import glob
import gzip
import io
import multiprocessing as mp
import os
import random
import sqlite3
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from itertools import islice, product
from pathlib import Path

import cartopy.crs as ccrs
import cv2
import geopandas as gpd
import mapbox_vector_tile
import mercantile
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from osgeo import gdal
from rasterio.errors import ShapeSkipWarning
from rasterio.features import rasterize, shapes
from PIL import Image
from scipy.ndimage import zoom
from shapely import (
    box,
    Polygon,
    make_valid,
)
from shapely.affinity import scale
from shapely.geometry import shape
from shapely.ops import unary_union

pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=ShapeSkipWarning)

gdal.UseExceptions()
gdal.PushErrorHandler("CPLQuietErrorHandler")

MAX_WORKERS = 10

ONAV_BASEMAP_DB = "/data/misc/shapes/onav_basemap_EPSG:32661.mbtiles"
ONAV_LAND_MASK_DB = "/data/misc/shapes/onav_land_mask_EPSG:32661.mbtiles"
ONAV_BATHY_DB = "/data/misc/shapes/onav_bathy_EPSG:32661.mbtiles"
ETOPO_PATH = "/data/misc/ETOPO2022/"
VRT_PATH = "/data/misc/etopo_global.vrt"
TILE_IMG_PATH = "/data/misc/shapes/tile_images/"

MBT_PATH = "/data/misc/maptiler-osm-2020-02-10-v3.11-planet.mbtiles"

TILE_SIZE = 256
TILE_BUFFER = 4
BUFFERED_SIZE = TILE_SIZE + (TILE_BUFFER * 2)

MAX_EXTENT = 20037508.342789244
WORLD_SIZE = MAX_EXTENT * 2

PROJ_MIN = -10637318.498257402
PROJ_MAX = 14637318.498257402
PROJ_EXTENT = PROJ_MAX - PROJ_MIN

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

UPS_NORTH = ccrs.Stereographic(
    central_latitude=90.0,
    central_longitude=0.0,
    false_easting=2000000.0,
    false_northing=2000000.0,
    scale_factor=0.994,
    globe=ccrs.Globe(datum="WGS84", ellipse="WGS84"),
)

PC_CRS = ccrs.PlateCarree()
WM_CRS = ccrs.epsg(3857)

contour_levels = np.array(
    [100, 200, 500, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 15000]
)
contour_levels = np.array([*contour_levels[::-1] * -1, *contour_levels])


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

    if row is None:
        return gpd.GeoDataFrame({"geometry": [], "class": []}, crs=3857)

    decompressed_data = gzip.decompress(row[0])
    decoded_tile = mapbox_vector_tile.decode(decompressed_data)

    del decompressed_data
    del row

    dfs = []

    ice_features = decoded_tile.get("landcover")
    if ice_features:
        filtered_ice = [
            f
            for f in ice_features["features"]
            if f.get("properties", {}).get("class") == "ice"
        ]
        if filtered_ice:
            df = gpd.GeoDataFrame.from_features(filtered_ice, crs=3857)
            df["class"] = "ice"
            dfs.append(df)

    water_features = decoded_tile.get("water")
    if water_features and water_features["features"]:
        df = gpd.GeoDataFrame.from_features(water_features["features"], crs=3857)
        df["class"] = "water"
        dfs.append(df)

    if not dfs:
        return gpd.GeoDataFrame({"geometry": [], "class": []}, crs=3857)

    tile_df = pd.concat(dfs, ignore_index=True).explode(ignore_index=True)
    return tile_df


def extract_mbt_data(x, y, z) -> gpd.GeoDataFrame:
    x_max = 2**z - 1

    tile_df = query_mbt(x, y, z)

    if len(tile_df) == 0:
        return tile_df

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

    invalid_idx = tile_df.loc[~tile_df.is_valid].index
    for idx in invalid_idx:
        tile_df.loc[idx, "geometry"] = make_valid(
            tile_df.loc[idx, "geometry"], method="structure", keep_collapsed=False
        )

    def optimize_geometries(df):
        optimized_rows = []
        for geom_class, group in df.groupby("class"):
            unionized = unary_union(group.geometry.values)
            optimized_rows.append({"class": geom_class, "geometry": unionized})
        return gpd.GeoDataFrame(optimized_rows, crs=3857)

    tile_df = optimize_geometries(tile_df)

    tile_df.geometry = tile_df.intersection(
        box(
            -TILE_BUFFER,
            -TILE_BUFFER,
            TILE_SIZE + TILE_BUFFER,
            TILE_SIZE + TILE_BUFFER,
        ),
    ).translate(xoff=TILE_BUFFER, yoff=TILE_BUFFER)

    invalid_idx = tile_df.loc[~tile_df.is_valid].index
    for idx in invalid_idx:
        tile_df.loc[idx, "geometry"] = make_valid(
            tile_df.loc[idx, "geometry"], method="structure", keep_collapsed=False
        )

    return tile_df


def create_tile_mask(x, y, z) -> np.ndarray:

    x_min, y_min, x_max, y_max = get_tile_bounds(x, y, z)

    tile_bounds = PC_CRS.transform_points(
        UPS_NORTH,
        np.array([x_min, x_min, x_max, x_max]),
        np.array([y_min, y_max, y_max, y_min]),
    )

    tile_x_min = tile_bounds[:, 0].min()
    tile_x_max = tile_bounds[:, 0].max()
    tile_y_min = tile_bounds[:, 1].min()
    tile_y_max = tile_bounds[:, 1].max()

    bbox = (tile_x_min, tile_y_min, tile_x_max, tile_y_max)
    zoom_levels = [z + 1]

    mbtiles_indices = []

    for tile in mercantile.tiles(*bbox, zooms=zoom_levels):
        mbtiles_indices.append([tile.x, tile.y, tile.z])

    n_pole_bounds = WM_CRS.transform_points(
        PC_CRS,
        np.array([-180, -90, 0, 90, 180, 180]),
        np.array([85, 85, 85, 85, 85, 90]),
    )

    n_pole_x_min = n_pole_bounds[:, 0].min()
    n_pole_x_max = n_pole_bounds[:, 0].max()
    n_pole_y_min = n_pole_bounds[:, 1].min()
    n_pole_y_max = n_pole_bounds[:, 1].max()

    np_geom = Polygon(n_pole_bounds[:, :2])

    np_df = gpd.GeoDataFrame(
        {"class": ["water"], "subclass": [None], "geometry": [np_geom]}, crs=3857
    )

    mbt_dfs = [np_df]

    for mbt_idx in mbtiles_indices:
        mbt_data = query_mbt(*mbt_idx)
        mbt_data.geometry = mbt_data.geometry.transform(
            lambda coords: mbt_to_wm(coords, *mbt_idx)
        )

        mbt_dfs.append(mbt_data)

    mbt_df = pd.concat(mbt_dfs, ignore_index=True)
    mbt_df = mbt_df.set_crs("EPSG:3857").to_crs(epsg=32661)

    invalid_idx = mbt_df.loc[~mbt_df.is_valid].index
    for idx in invalid_idx:
        mbt_df.loc[idx, "geometry"] = make_valid(
            mbt_df.loc[idx, "geometry"], method="structure", keep_collapsed=False
        )

    mbt_clipped = mbt_df.clip(box(x_min, y_min, x_max, y_max))

    invalid_idx = mbt_clipped.loc[~mbt_clipped.is_valid].index
    for idx in invalid_idx:
        mbt_clipped.loc[idx, "geometry"] = make_valid(
            mbt_clipped.loc[idx, "geometry"], method="structure", keep_collapsed=False
        )

    mbt_clipped = mbt_clipped.dissolve(by="class").reset_index().explode()

    mbt_px = mbt_clipped.copy()
    mbt_px.geometry = mbt_clipped.transform(
        lambda coords: ups_n_to_tile_coords(coords, x, y, z)
    )

    water_mask = rasterize(
        mbt_px.loc[mbt_px["class"] == "water", "geometry"].explode(),
        out_shape=(BUFFERED_SIZE, BUFFERED_SIZE),
        all_touched=True,
    )

    ice_mask = rasterize(
        mbt_px.loc[mbt_px["class"] == "ice", "geometry"].explode(),
        out_shape=(BUFFERED_SIZE, BUFFERED_SIZE),
        all_touched=True,
    )

    tile_mask = np.ones((BUFFERED_SIZE, BUFFERED_SIZE), dtype=int)
    tile_mask[water_mask == 1] = 0
    tile_mask[ice_mask == 1] = 2

    return np.flipud(tile_mask)


def create_vrt():
    if os.path.exists(VRT_PATH):
        return

    all_nc_files = glob.glob("/data/misc/ETOPO2022/15s/15s_surface_elev_netcdf/*.nc")
    subdatasets = [f'NETCDF:"{f}":z' for f in all_nc_files]

    vrt_ds = gdal.BuildVRT(VRT_PATH, subdatasets)

    if vrt_ds is not None:
        vrt_ds.FlushCache()
        vrt_ds = None
    else:
        print("GDAL failed to initialize the VRT dataset mapping.")


def get_topo_contours(x, y, z, block_size):
    b_left, b_bottom, b_right, b_top = get_tile_bounds(x, y, z)

    block_px_size = block_size * TILE_SIZE + 2 * TILE_BUFFER

    warp_width = block_px_size
    warp_height = block_px_size

    gdal.SetConfigOption("GDAL_HTTP_UNSUB_OK", "YES")

    try:
        ds = gdal.Warp(
            "",
            VRT_PATH,
            format="MEM",
            dstSRS="EPSG:32661",
            resampleAlg=gdal.GRIORA_Bilinear,
            outputBounds=[b_left, b_bottom, b_right, b_top],
            width=warp_width,
            height=warp_height,
            warpMemoryLimit=256 * 1024 * 1024,
        )

        band = ds.GetRasterBand(1)
        tile_data = band.ReadAsArray(buf_type=gdal.GDT_Int16)
        ds = None

        contour_mask = tile_data.copy()

        for l0, l1 in zip(contour_levels[:-1], contour_levels[1:]):
            contour_mask[(contour_mask >= l0) & (contour_mask < l1)] = l0

    except Exception as e:
        print(f"Error rendering edge tile {z}/{x}/{y}: {e}")
    finally:
        gdal.SetConfigOption("GDAL_HTTP_UNSUB_OK", None)

    return contour_mask


def mbt_to_wm(coords, x, y, z):

    tile_x = coords[:, 0]
    tile_y = 4096 - coords[:, 1]

    MAX_EXTENT = 20037508.342789244
    resolution = 2 * MAX_EXTENT / 4096 / (2**z)

    world_x = tile_x + x * 4096
    world_y = tile_y + y * 4096

    mx = (world_x * resolution) - MAX_EXTENT
    my = MAX_EXTENT - (world_y * resolution)

    return np.stack([mx, my]).T


def get_tile_bounds(x, y, z, buffered=True):
    n_tiles = 2**z
    tile_width = PROJ_EXTENT / n_tiles
    pixel_resolution = tile_width / TILE_SIZE

    left = PROJ_MIN + (x * tile_width)
    top = PROJ_MAX - (y * tile_width)
    right = left + tile_width
    bottom = top - tile_width

    if not buffered:
        return left, bottom, right, top

    buf_meters = TILE_BUFFER * pixel_resolution
    b_left, b_bottom, b_right, b_top = (
        left - buf_meters,
        bottom - buf_meters,
        right + buf_meters,
        top + buf_meters,
    )

    return b_left, b_bottom, b_right, b_top


def ups_n_to_tile_coords(coords, x, y, z):

    tile_x = coords[:, 0]
    tile_y = coords[:, 1]

    x_min, y_min, x_max, y_max = get_tile_bounds(x, y, z)

    px_x = (tile_x - x_min) / (x_max - x_min) * BUFFERED_SIZE
    px_y = (tile_y - y_min) / (y_max - y_min) * BUFFERED_SIZE

    px_x = np.round(px_x, decimals=0)
    px_y = np.round(px_y, decimals=0)

    return np.stack([px_x, px_y]).T


def mbt_to_wm(coords, x, y, z):

    tile_x = coords[:, 0]
    tile_y = 4096 - coords[:, 1]

    MAX_EXTENT = 20037508.342789244
    resolution = 2 * MAX_EXTENT / 4096 / (2**z)

    world_x = tile_x + x * 4096
    world_y = tile_y + y * 4096

    mx = (world_x * resolution) - MAX_EXTENT
    my = MAX_EXTENT - (world_y * resolution)

    return np.stack([mx, my]).T


def get_contour_mask(x, y, z, block_size):
    if z < 4:
        temp_x = np.arange(x * 2 ** (4 - z), (x + 1) * 2 ** (4 - z))
        temp_y = np.arange(y * 2 ** (4 - z), (y + 1) * 2 ** (4 - z))

        contours = np.zeros(
            (
                TILE_SIZE * 2 ** (4 - z) + 2 * TILE_BUFFER,
                TILE_SIZE * 2 ** (4 - z) + 2 * TILE_BUFFER,
            )
        )

        for x_idx, t_x in enumerate(temp_x):
            for y_idx, t_y in enumerate(temp_y):
                sub_contour = get_topo_contours(t_x, t_y, 4, 1)

                x_start = x_idx * TILE_SIZE
                if x_idx > 0:
                    x_start -= TILE_BUFFER
                x_end = x_start + BUFFERED_SIZE

                y_start = y_idx * TILE_SIZE
                if y_idx > 0:
                    y_start -= TILE_BUFFER
                y_end = y_start + BUFFERED_SIZE

                contours[y_start:y_end, x_start:x_end] = sub_contour

        target_shape = (BUFFERED_SIZE, BUFFERED_SIZE)
        zoom_factors = (
            target_shape[0] / contours.shape[0],
            target_shape[1] / contours.shape[1],
        )
        contour_mask = zoom(contours, zoom_factors, order=0)
    else:
        contour_mask = get_topo_contours(x, y, z, block_size)


def create_tile(x, y, z, contour_mask) -> None:
    x = int(x)
    y = int(y)
    z = int(z)

    contour_mask = contour_mask.astype(np.float32)

    print(f"Starting tile ({x}, {y}, {z}).")
    tile_mask = create_tile_mask(x, y, z)

    tile_img = np.zeros(
        (TILE_SIZE + 2 * TILE_BUFFER, TILE_SIZE + 2 * TILE_BUFFER, 3), dtype=int
    )

    for layer_type, mask_value in zip(["bathy", "land", "ice"], [0, 1, 2]):
        if ~np.any(tile_mask == mask_value):
            continue

        for level in contour_levels:
            level_color = TILE_COLORS[layer_type].get(level)
            if level_color is None:
                level_color = TILE_COLORS[layer_type][0]

            r, g, b = bytes.fromhex(level_color)
            tile_img[(tile_mask == mask_value) & (contour_mask == level)] = [r, g, b]

    tile_img = tile_img.astype(np.uint8)
    tile_img = cv2.inpaint(
        tile_img,
        (255 * np.all(tile_img == 0, axis=-1)).astype(np.uint8),
        3,
        cv2.INPAINT_TELEA,
    )

    land_mask = tile_mask.copy()

    contour_mask[tile_mask != 0] = np.nan

    water_mask_shapes = shapes(
        tile_mask,
        mask=(tile_mask == 0),
        transform=rasterio.Affine.identity(),
    )

    water_mask_shapes = shapes(
        contour_mask,
        mask=(tile_mask == 0),
        transform=rasterio.Affine.identity(),
    )

    water_mask_polygons = []
    for geom, value in water_mask_shapes:
        water_mask_polygons.append(shape(geom))

    bathy_contours = gpd.GeoDataFrame(geometry=water_mask_polygons, crs=3857)

    all_boundaries = bathy_contours.geometry.boundary.union_all()
    outer_perimeter = box(
        0, 0, TILE_SIZE + 2 * TILE_BUFFER, TILE_SIZE + 2 * TILE_BUFFER
    ).boundary
    internal_borders = all_boundaries.difference(outer_perimeter)
    bathy_contours = gpd.GeoDataFrame(
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

    land_mask[land_mask == 2] = 1
    land_mask = land_mask[TILE_BUFFER:-TILE_BUFFER, TILE_BUFFER:-TILE_BUFFER]
    tile_img = tile_img[TILE_BUFFER:-TILE_BUFFER, TILE_BUFFER:-TILE_BUFFER, :]

    land_mask_blob = gzip.compress(land_mask.astype(np.uint8).tobytes())

    write_tile_blob(y, x, z, land_mask_blob, ONAV_LAND_MASK_DB)

    image_path = f"{TILE_IMG_PATH}onav_{x}_{y}_{z}.png"

    img = Image.fromarray(tile_img)
    img.save(image_path)

    with open(image_path, "rb") as file:
        image_blob = file.read()
        write_tile_blob(y, x, z, image_blob, ONAV_BASEMAP_DB)

    print(f"Completed tile ({x}, {y}, {z}).")


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


def filter_tile_group(tile_group):
    x = tile_group[:, 0]
    y = tile_group[:, 1]
    z = tile_group[:, 2]

    x_min = int(x.min())
    x_max = int(x.max())
    y_min = int(y.min())
    y_max = int(y.max())

    zoom = z[0]

    filtered = tile_group

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

        if len(tile_coords) > 0:
            mask = (tile_group[:, None, :2] == tile_coords).all(axis=2).any(axis=1)
            filtered = tile_group[~mask]

    except Exception as e:
        print(e)

    return filtered


def get_block_size(z: int) -> int:
    z = int(z)
    block_size = 2 ** (z - 7)
    if block_size < 1:
        block_size = 1
    return int(block_size)


def process_tile_block(block_x, block_y, z) -> None:

    block_size = get_block_size(z)

    x_min = int(block_x * block_size)
    x_max = int(x_min + block_size)

    y_min = int(block_y * block_size)
    y_max = int(y_min + block_size)

    block_tiles = product(range(x_min, x_max), range(y_min, y_max))

    tile_block_arr = np.array([[x, y, z] for x, y in block_tiles])

    filtered_tiles = filter_tile_group(tile_block_arr)

    if len(filtered_tiles) == 0:
        print(
            f"Tiles ({x_min}, {y_min}, {z}) - ({x_max}, {y_max}, {z}) previously completed."
        )
        return

    contour_mask = get_contour_mask(x_min, y_min, z, block_size)

    contour_masks = [
        contour_mask[
            (y - y_min) * TILE_SIZE : (y + 1 - y_min) * TILE_SIZE + 2 * TILE_BUFFER,
            (x - x_min) * TILE_SIZE : (x + 1 - x_min) * TILE_SIZE + 2 * TILE_BUFFER,
        ]
        for x, y, _ in filtered_tiles
    ]

    if len(filtered_tiles) > 0:
        for filtered_tile in filtered_tiles:
            x, y, z = filtered_tile
            contour_mask = contour_masks.pop(0)
            create_tile(x, y, z, contour_mask)


def process_tiles(z) -> None:
    print(f"Starting zoom {z}")

    n_tiles = 2**z
    block_size = get_block_size(z)

    n_blocks = int(n_tiles / block_size)

    block_indexes = product(range(n_blocks), range(n_blocks))

    max_workers = z
    if max_workers > MAX_WORKERS:
        max_workers = MAX_WORKERS

    chunk_size = 64

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        while True:
            block_chunk = list(islice(block_indexes, chunk_size))
            if not block_chunk:
                break

            block_chunk_arr = np.array([[x, y, z] for x, y in block_chunk])

            executor.map(
                process_tile_block,
                block_chunk_arr[:, 0],
                block_chunk_arr[:, 1],
                block_chunk_arr[:, 2],
            )

            gc.collect()
        gc.collect()
    gc.collect()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    sqlite3.register_adapter(np.ndarray, adapt_array)
    sqlite3.register_converter("array", convert_array)

    if not os.path.exists(ONAV_BASEMAP_DB):
        create_mbt_db(ONAV_BASEMAP_DB)

    if not os.path.exists(ONAV_LAND_MASK_DB):
        create_mbt_db(ONAV_LAND_MASK_DB)

    if not os.path.exists(ONAV_BATHY_DB):
        create_mbt_db(ONAV_BATHY_DB)

    create_vrt()

    for z in range(1, 15):
        process_tiles(z)
