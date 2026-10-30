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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from itertools import islice, product, repeat
from pathlib import Path

import cartopy.crs as ccrs
import cv2
import geopandas as gpd
import mapbox_vector_tile
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from osgeo import gdal
from rasterio.errors import ShapeSkipWarning
from rasterio.features import rasterize, shapes
from PIL import Image
from shapely import (
    box,
    Polygon,
    make_valid,
    MultiPolygon,
)
from shapely.affinity import scale
from shapely.geometry import shape
from shapely.ops import unary_union

pd.options.mode.chained_assignment = None
warnings.filterwarnings("ignore", category=ShapeSkipWarning)

gdal.UseExceptions()
gdal.PushErrorHandler("CPLQuietErrorHandler")

MAX_WORKERS = 8

ONAV_BASEMAP_DB = "/data/misc/shapes/onav_basemap.mbtiles"
ONAV_LAND_MASK_DB = "/data/misc/shapes/onav_land_mask.mbtiles"
ONAV_BATHY_DB = "/data/misc/shapes/onav_bathy.mbtiles"
ETOPO_PATH = "/data/misc/ETOPO2022/"
VRT_PATH = "/data/misc/etopo_global.vrt"
TILE_IMG_PATH = "/data/misc/shapes/tile_images/"

MBT_PATH = "/data/misc/maptiler-osm-2020-02-10-v3.11-planet.mbtiles"

TILE_SIZE = 256
TILE_BUFFER = 4
BUFFERED_SIZE = TILE_SIZE + (TILE_BUFFER * 2)

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
                if len(ext) >= 4:
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
        contour_df = pd.read_pickle(contour_path)
    else:
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

    return contour_df


def etopo_px_contours(etopo_file: str | Path, zoom: int) -> None:
    if not isinstance(etopo_file, Path):
        etopo_file = Path(etopo_file)

    etopo_idx = etopo_file.name.split("ETOPO_2022_v1_15s_")[-1].split("_surface.nc")[0]

    px_contour_path = Path(
        f"/data/misc/ETOPO_contours/z{zoom}/contour_px_{etopo_idx}_EPSE3857_z{zoom}.pkl"
    )

    if not px_contour_path.is_file():
        px_contour_path.parent.mkdir(parents=True, exist_ok=True)

        etopo_df = extract_etopo_contours(etopo_file)

        etopo_df.geometry = etopo_df.transform(
            lambda xy: wm_to_tile_coords(xy, zoom)
        ).make_valid(method="structure", keep_collapsed=False)

        etopo_df.to_pickle(px_contour_path)

    else:
        etopo_df = pd.read_pickle(px_contour_path)

    return etopo_df


def create_tile_mask(x, y, z) -> np.ndarray:

    mbt_df = extract_mbt_data(x, y, z)

    tile_shape = (TILE_SIZE + 2 * TILE_BUFFER, TILE_SIZE + 2 * TILE_BUFFER)

    tile_mask = np.ones(tile_shape)

    if len(mbt_df) > 0:
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
    n_tiles = 2**z
    tile_width = (MAX_EXTENT * 2) / n_tiles
    pixel_resolution = tile_width / TILE_SIZE

    left = -MAX_EXTENT + (x * tile_width)
    top = MAX_EXTENT - (y * tile_width)
    right = left + tile_width * block_size
    bottom = top - tile_width * block_size

    buf_meters = TILE_BUFFER * pixel_resolution
    b_left, b_bottom, b_right, b_top = (
        left - buf_meters,
        bottom - buf_meters,
        right + buf_meters,
        top + buf_meters,
    )

    c_left = max(b_left, -MAX_EXTENT)
    c_bottom = max(b_bottom, -MAX_EXTENT)
    c_right = min(b_right, MAX_EXTENT)
    c_top = min(b_top, MAX_EXTENT)

    pad_left = int(round((c_left - b_left) / pixel_resolution))
    pad_right = int(round((b_right - c_right) / pixel_resolution))
    pad_bottom = int(round((c_bottom - b_bottom) / pixel_resolution))
    pad_top = int(round((b_top - c_top) / pixel_resolution))

    block_px_size = block_size * TILE_SIZE + 2 * TILE_BUFFER

    warp_width = block_px_size - pad_left - pad_right
    warp_height = block_px_size - pad_top - pad_bottom

    gdal.SetConfigOption("GDAL_HTTP_UNSUB_OK", "YES")

    try:
        ds = gdal.Warp(
            "",
            VRT_PATH,
            format="MEM",
            dstSRS="EPSG:3857",
            resampleAlg=gdal.GRIORA_Bilinear,
            outputBounds=[c_left, c_bottom, c_right, c_top],
            width=warp_width,
            height=warp_height,
        )

        band = ds.GetRasterBand(1)
        tile_data = band.ReadAsArray()
        ds = None

        padded_tile_data = np.pad(
            tile_data,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=-32768,
        )

        contour_mask = padded_tile_data.copy()

        for l0, l1 in zip(contour_levels[:-1], contour_levels[1:]):
            contour_mask[(contour_mask >= l0) & (contour_mask < l1)] = l0

    except Exception as e:
        print(f"Error rendering edge tile {z}/{x}/{y}: {e}")
    finally:
        gdal.SetConfigOption("GDAL_HTTP_UNSUB_OK", None)

    return contour_mask


def create_tile(x, y, z, contour_mask) -> None:
    x = int(x)
    y = int(y)
    z = int(z)

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

    write_tile_blob(y, x, z, land_mask.astype(np.uint8), ONAV_LAND_MASK_DB)

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


def process_tile_block(block_x, block_y, z) -> None:

    block_size = 2 ** (z - 8)
    if block_size < 1:
        block_size = 1

    x_min = int(block_x * block_size)
    x_max = int(x_min + block_size)

    y_min = int(block_y * block_size)
    y_max = int(y_min + block_size)

    block_tiles = product(range(x_min, x_max), range(y_min, y_max))

    tile_block_arr = np.array([[x, y, z] for x, y in block_tiles])

    filtered_tiles = filter_tile_group(tile_block_arr)

    if len(filtered_tiles) == 0:
        print(f"Tiles ({x_min}, {y_min}, {z}) - ({x_max}, {y_max}, {z}) previously completed.")
        return

    contour_mask = get_topo_contours(x_min, y_min, z, block_size)
    if x_min == 0:
        wrap_contours = get_topo_contours(2**z - block_size - 1, y_min, z, block_size)
        contour_mask[:, :TILE_BUFFER] = wrap_contours[
            :, -TILE_BUFFER * 2 : -TILE_BUFFER
        ]

    if x_max == 2**z - 1:
        wrap_contours = get_topo_contours(0, y_min, z, block_size)
        contour_mask[:, -TILE_BUFFER:] = wrap_contours[:, TILE_BUFFER : TILE_BUFFER * 2]

    contour_masks = [
        contour_mask[
            (x - x_min) * TILE_SIZE : (x + 1 - x_min) * TILE_SIZE + 2 * TILE_BUFFER,
            (y - y_min) * TILE_SIZE : (y + 1 - y_min) * TILE_SIZE + 2 * TILE_BUFFER,
        ]
        for x, y, _ in filtered_tiles
    ]

    with ThreadPoolExecutor() as thread_exec:
        if len(filtered_tiles) > 0:
            thread_exec.map(
                create_tile,
                filtered_tiles[:, 0],
                filtered_tiles[:, 1],
                repeat(z),
                contour_masks,
            )


def process_tiles(z) -> None:
    n_tiles = 2**z
    block_size = 2 ** (z - 8)
    if block_size < 1:
        block_size = 1

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

    for z in range(12, 15):
        process_tiles(z)
