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
import xarray as xr
from PIL import Image
from rasterio import features
from shapely import (
    box,
    LineString,
    Point,
    Polygon,
    MultiPolygon,
)
from shapely.affinity import scale

ONAV_BASEMAP_DB = "/data/misc/shapes/onav_basemap.mbtiles"
ONAV_LAND_MASK_DB = "/data/misc/shapes/onav_land_mask.mbtiles"
ONAV_BATHY_DB = "/data/misc/shapes/onav_bathy.mbtiles"
OSM_LAND_POLY_PATH = "/data/misc/osm/land-polygons-complete-3857/land_polygons.shp"
ETOPO_PATH = "/data/misc/ETOPO2022/"
OSM_PATH = "/data/misc/osm/"
TILE_IMG_PATH = "/data/misc/shapes/tile_images/"

pc_crs = ccrs.PlateCarree()
wm_crs = ccrs.epsg("3857")

wm_radius = 20037508.34
max_zoom = 14
tile_res = 256
px_buffer = 8

tile_colors = {
    "coastline": {
        0: "f6e8c3",
    },
    "land": {
        0: "f6e8c3",
        1: "f0deb0",
        2: "e8d29a",
        3: "e1c684",
        4: "d8b46c",
        5: "cd9d50",
        6: "c28633",
        7: "af7222",
        8: "9b5f14",
        9: "844c09",
        10: "6c3e07",
        11: "543005",
    },
    "bathy": {
        0: "8a8989",
        1: "7e7d7d",
        2: "737372",
        3: "6a6a69",
        4: "626161",
        5: "5a5a5a",
        6: "535352",
        7: "4c4b4b",
        8: "474646",
        9: "414140",
        10: "3c3b3b",
        11: "373737",
    },
    "ice": {
        0: "8683bd",
        1: "8d89c0",
        2: "9390c3",
        3: "9a96c6",
        4: "a19eca",
        5: "aaa8d0",
        6: "b2b2d5",
        7: "bcbddc",
        8: "c5c6e1",
        9: "cfd0e6",
        10: "dadaeb",
        11: "e2e2ef",
    },
}

levels = np.array([100, 200, 500, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 15000])

osm_paths = [
    "/data/misc/osm/africa-latest.osm.pbf",
    "/data/misc/osm/antarctica-latest.osm.pbf",
    "/data/misc/osm/asia-latest.osm.pbf",
    "/data/misc/osm/australia-oceania-latest.osm.pbf",
    "/data/misc/osm/central-america-latest.osm.pbf",
    "/data/misc/osm/europe-latest.osm.pbf",
    "/data/misc/osm/north-america-latest.osm.pbf",
    "/data/misc/osm/south-america-latest.osm.pbf",
]

poly_paths = [
    "/data/misc/osm/africa.poly",
    "/data/misc/osm/antarctica.poly",
    "/data/misc/osm/asia.poly",
    "/data/misc/osm/australia-oceania.poly",
    "/data/misc/osm/central-america.poly",
    "/data/misc/osm/europe.poly",
    "/data/misc/osm/north-america.poly",
    "/data/misc/osm/south-america.poly",
]


def find_contours(surface: np.array) -> np.array:
    levels = np.array([100, 200, 500, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 15000])

    contours = surface.copy()
    contours[~np.isnan(contours)] = 0

    for idx, level in enumerate(levels[:-1]):
        contours[surface >= level] = contours[surface >= level] + 1

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


def get_contour_polys(contours, x_coords, y_coords, layer_type):
    contour_df = gpd.GeoDataFrame(
        columns=["geometry", "level", "layer_type"],
        geometry="geometry",
        crs=3857,
    )
    for level in range(14):
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
                    "layer_type": [layer_type],
                },
                crs=3857,
            )
            level_df.geometry = level_df.make_valid(
                method="structure", keep_collapsed=False
            )
            contour_df = pd.concat([contour_df, level_df], ignore_index=True)
    return contour_df


def parse_poly_to_shapely(filepath):
    coords = []
    polys = []
    with open(filepath, "r") as f:
        lines = f.readlines()
        for line in lines:
            line_data = line.split()
            if len(line_data) == 2:
                coords.append([float(p) for p in line_data])
            elif "END" in line and len(coords) > 0:
                polys.append(Polygon(coords))
                coords = []

    return MultiPolygon(polys)


def split_osm_data(etopo_file) -> None:
    if not isinstance(etopo_file, Path):
        etopo_file = Path(etopo_file)

    etopo_ds = xr.open_dataset(etopo_file)

    lat = etopo_ds.lat.data
    lon = etopo_ds.lon.data

    x0 = int(np.floor(lon.min()))
    y0 = int(np.floor(lat.min()))
    x1 = int(np.ceil(lon.max()))
    y1 = int(np.ceil(lat.max()))

    if x0 < -180:
        x0 = -180
    if y0 < -90:
        y0 = -90
    if x1 > 180:
        x1 = 180
    if y1 > 90:
        y1 = 90

    etopo_bounds = box(x0, y0, x1, y1)

    for osm_path, poly_path in zip(osm_paths, poly_paths):
        osm_bounds = parse_poly_to_shapely(poly_path)

        if osm_bounds.intersects(etopo_bounds):
            bbox_str = f"{x0},{y0},{x1},{y1}"
            etopo_idx = (
                "_"
                + etopo_file.name.split("ETOPO_2022_v1_15s_")[-1].split("_surface.nc")[
                    0
                ]
            )
            bbox_output_path = osm_path.split(".")[0] + etopo_idx + ".osm.pbf"
            natural_output_path = (
                osm_path.split(".")[0] + etopo_idx + "_natural.osm.pbf"
            )

            if not os.path.exists(bbox_output_path):
                subprocess.run(
                    [
                        "osmium",
                        "extract",
                        "-b",
                        f"{bbox_str}",
                        f"{osm_path}",
                        "-o",
                        f"{bbox_output_path}",
                        "--strategy",
                        "complete_ways",
                    ]
                )
            if not os.path.exists(natural_output_path):
                subprocess.run(
                    [
                        "osmium",
                        "tags-filter",
                        f"{bbox_output_path}",
                        "nwr/natural=water",
                        "nwr/waterway=riverbank",
                        "-o",
                        f"{natural_output_path}",
                    ]
                )


def read_osm_land_polys(wm_bbox: Polygon) -> gpd.GeoDataFrame:
    land_df = gpd.read_file(
        "/data/misc/osm/land-polygons-complete-3857/land_polygons.shp",
        bbox=tuple(wm_bbox.bounds),
    )
    land_df = land_df.set_geometry(land_df.geometry.intersection(wm_bbox))
    land_df = land_df[~land_df.geometry.is_empty]
    land_df.geometry = land_df.make_valid(method="structure", keep_collapsed=False)
    land_df.drop(columns=["FID"], inplace=True)
    land_df["level"] = 0
    land_df["layer_type"] = "coastline"
    land_df = land_df.dissolve(by=["level", "layer_type"], as_index=False)

    return land_df


def get_osm_data(tile_id: str) -> gpd.GeoDataFrame:
    osm_files = glob.glob(f"/data/misc/osm/*{tile_id}_natural.osm.pbf")

    osm_geoms = []
    for osm_file in osm_files:
        temp_df = gpd.read_file(
            osm_file,
            layer="multipolygons",
            engine="pyogrio",
        )
        temp_df = temp_df.to_crs(3857)
        temp_df = temp_df[["osm_id", "natural", "other_tags", "geometry"]]
        temp_df = temp_df[
            (temp_df["natural"] == "water")
            & (
                temp_df["other_tags"].str.contains('"water"=>"river"')
                | temp_df["other_tags"].str.contains('"water"=>"lake"')
                | temp_df["other_tags"].str.contains('"water"=>"pond"')
                | temp_df["other_tags"].str.contains('"water"=>"canal"')
            )
        ]
        temp_df.geometry = temp_df.make_valid(method="structure", keep_collapsed=False)
        osm_geoms.append(temp_df.union_all())

    osm_df = gpd.GeoDataFrame({"geometry": osm_geoms}, crs=3857)

    return osm_df


def get_coastline_polys(wm_bbox: np.ndarray, tile_id: str) -> gpd.GeoDataFrame:
    land_df = read_osm_land_polys(wm_bbox)

    if len(land_df) > 0:
        osm_df = get_osm_data(tile_id)

        land_df.geometry = land_df.difference(osm_df)

        if land_df.geom_type[0] != "MultiPolygon":
            land_df = land_df.explode(ignore_index=True)
            land_df = land_df[
                (land_df.geom_type == "Polygon") | (land_df.geom_type == "MultiPolygon")
            ]
            land_df = land_df.dissolve()

    return land_df


def extract_etopo_contours(etopo_file: str | Path) -> None:
    if not isinstance(etopo_file, Path):
        etopo_file = Path(etopo_file)

    etopo_idx = etopo_file.name.split("ETOPO_2022_v1_15s_")[-1].split("_surface.nc")[0]

    contour_path = f"/data/misc/ETOPO_contours/contour_{etopo_idx}.pkl"
    if os.path.exists(contour_path):
        contour_df = pd.read_pickle(contour_path)
    else:
        print(f"Extracting contours from {etopo_file}")
        etopo_ds = xr.open_dataset(etopo_file)

        lat = etopo_ds.lat.data
        lon = etopo_ds.lon.data

        lon_grid, lat_grid = np.meshgrid(lon, lat)
        wm_coords = wm_crs.transform_points(pc_crs, lon_grid, lat_grid)[:, :, :2]
        wm_x_coords = wm_coords[:, :, 0]
        wm_y_coords = wm_coords[:, :, 1]

        wm_bbox = box(
            wm_x_coords.min(), wm_y_coords.min(), wm_x_coords.max(), wm_y_coords.max()
        )

        coastline_df = get_coastline_polys(wm_bbox, etopo_idx)

        land_contours = find_contours(etopo_ds.z.values)
        bathy_contours = find_contours(etopo_ds.z.values * -1)

        bed_file = f"/data/misc/ETOPO2022/15s/15s_bed_elev_netcdf/ETOPO_2022_v1_15s_{etopo_idx}_bed.nc"
        if os.path.exists(bed_file):
            bed_ds = xr.open_mfdataset(bed_file)

            ice_mask = etopo_ds.z.values - bed_ds.z.values
            ice_mask[etopo_ds.z.values < 0] = np.nan
            ice_mask[bed_ds.z.values > etopo_ds.z.values] = np.nan
            ice_mask[ice_mask < 10] = np.nan
            ice_mask[~np.isnan(ice_mask)] = 1
            ice_mask[ice_mask != 1] = 0

            ice_contours = land_contours.copy()

            land_contours[ice_mask == 1] = np.nan
            ice_contours[ice_mask == 0] = np.nan

            ice_polys = get_contour_polys(ice_contours, wm_x_coords, wm_y_coords, "ice")
        else:
            ice_polys = gpd.GeoDataFrame(
                columns=["geometry", "level", "layer_type"],
                geometry="geometry",
                crs=3857,
            )

        land_polys = get_contour_polys(land_contours, wm_x_coords, wm_y_coords, "land")
        bathy_polys = get_contour_polys(
            bathy_contours, wm_x_coords, wm_y_coords, "bathy"
        )

        contour_df = pd.concat(
            [bathy_polys, coastline_df, land_polys, ice_polys], ignore_index=True
        )
        layer_order = ["bathy", "coastline", "land", "ice"]
        contour_df["layer_type"] = pd.Categorical(
            contour_df["layer_type"], categories=layer_order, ordered=True
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

    px_contour_path = (
        f"/data/misc/ETOPO_contours/contour_px_{etopo_idx}_EPSE3857_z{zoom}.pkl"
    )

    if not os.path.exists(px_contour_path):

        etopo_df = extract_etopo_contours(etopo_file)

        etopo_df.geometry = etopo_df.transform(
            lambda xy: wm_to_tile_coords(xy, zoom)
        ).make_valid(method="structure", keep_collapsed=False)

        etopo_df.to_pickle(px_contour_path)

    else:
        etopo_df = pd.read_pickle(px_contour_path)

    return etopo_df

def wm_to_tile_coords(wm_coords, zoom) -> np.ndarray:
    wm_x = wm_coords[:, 0]
    wm_y = wm_coords[:, 1]

    n_tiles = 2**zoom

    px_x = tile_res * n_tiles * (wm_x + wm_radius) / (2 * wm_radius)
    px_y = tile_res * n_tiles * (wm_y + wm_radius) / (2 * wm_radius)

    px_coords = np.vstack([px_x.astype(int), px_y.astype(int)]).T

    return np.array(px_coords, dtype=np.float64)


def get_etopo_idx(etopo_file: str) -> str:
    etopo_idx = etopo_file.split("ETOPO_2022_v1_15s_")[-1].split("_surface.nc")[0]

    return etopo_idx


def etopo_grid_coords() -> gpd.GeoDataFrame:
    etopo_files = sorted(
        glob.glob("/data/misc/ETOPO2022/15s/15s_surface_elev_netcdf/*.nc")
    )

    idx_coords = []
    for etopo_file in etopo_files:
        etopo_idx = get_etopo_idx(etopo_file)
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


def write_onav_tiles(contour_px_df, tile_x, tile_y, zoom):
    print(f"Writing tile ({tile_x}, {tile_y}, {zoom}).")
    tile_px_bounds = np.array(
        [
            tile_x * tile_res - px_buffer,
            tile_y * tile_res - px_buffer,
            (tile_x + 1) * tile_res + px_buffer,
            (tile_y + 1) * tile_res + px_buffer,
        ]
    )

    tile_contours = contour_px_df.clip(tile_px_bounds).sort_index()
    tile_contours.geometry = tile_contours.make_valid(
        method="structure", keep_collapsed=False
    )

    tile_contours.geometry = tile_contours.translate(
        xoff=-tile_px_bounds[0], yoff=-tile_px_bounds[1]
    )

    land_mask = features.rasterize(
        tile_contours.loc[tile_contours["layer_type"] == "coastline"]
        .explode()
        .geometry,
        out_shape=((tile_res, tile_res)),
        all_touched=True,
    )

    tile_img = np.zeros((tile_res, tile_res, 3), dtype=int)
    for _, row in tile_contours.iterrows():
        if row.geometry.geom_type == "MultiPolygon":
            row_geom = row.geometry.geoms
        else:
            row_geom = [row.geometry]
        layer_mask = features.rasterize(
            row_geom,
            out_shape=(tile_res, tile_res),
            all_touched=True,
        )
        r, g, b = bytes.fromhex(tile_colors[row.layer_type][row.level])
        mask_idx = 0
        if row.layer_type != "bathy":
            mask_idx = 1

        tile_img[(layer_mask == 1) & (land_mask == mask_idx)] = [r, g, b]

    tile_img = tile_img.astype(np.uint8)
    tile_img = cv2.inpaint(
        tile_img,
        (255 * np.all(tile_img == 0, axis=-1)).astype(np.uint8),
        3,
        cv2.INPAINT_TELEA,
    )

    land_mask = np.flipud(land_mask)
    tile_img = np.flipud(tile_img)

    image_path = f"{TILE_IMG_PATH}onav_{tile_x}_{tile_y}_{zoom}.png"

    img = Image.fromarray(tile_img)
    img.save(image_path)

    with open(image_path, "rb") as file:
        image_blob = file.read()
        write_tile_blob(tile_y, tile_x, zoom, image_blob, ONAV_BASEMAP_DB)

    write_tile_blob(tile_y, tile_x, zoom, land_mask, ONAV_LAND_MASK_DB)

    bathy_df = tile_contours.loc[
        (tile_contours.level > 0) & (tile_contours.layer_type == "bathy")
    ].explode(ignore_index=True)
    bathy_df = bathy_df[
        (bathy_df.geom_type == "Polygon") | (bathy_df.geom_type == "MultiPolygon")
    ]

    bathy_df.geometry = bathy_df.geometry.boundary.buffer(0.5)
    bathy_df = gpd.GeoDataFrame({"geometry": bathy_df.explode().exterior})
    bathy_df.geometry = bathy_df.geometry.apply(lambda geom: LineString(geom))
    bathy_df.geometry = bathy_df.geometry.apply(
        lambda geom: scale(
            geom,
            xfact=4096 / tile_res,
            yfact=4096 / tile_res,
            origin=(0, 0),
        )
    )
    if len(bathy_df) > 0:
        write_vector_tile(
            tile_y,
            tile_x,
            zoom,
            bathy_df.geometry,
            ONAV_BATHY_DB,
        )


def process_etopo(etopo_file: str, zoom: int) -> None:
    print(f"Processing ETOPO file {etopo_file} zoom level {zoom}.")

    etopo_coords = etopo_grid_coords()

    etopo_file_data = etopo_coords.loc[etopo_coords["etopo_file"] == etopo_file]
    etopo_file_data.geometry = etopo_file_data.transform(
        lambda xy: wm_to_tile_coords(xy, zoom)
    )

    tile_x_min, tile_y_min, tile_x_max, tile_y_max = etopo_file_data.total_bounds / tile_res

    tile_x_min = int(np.ceil(tile_x_min))
    tile_y_min = int(np.ceil(tile_y_min))
    tile_x_max = int(np.ceil(tile_x_max))
    tile_y_max = int(np.ceil(tile_y_max))

    if tile_x_min < 0:
        tile_x_min = 0
    if tile_y_min < 0:
        tile_y_min = 0
    if tile_x_max > 2**zoom - 1:
        tile_x_max = 2**zoom - 1
    if tile_y_max > 2**zoom - 1:
        tile_y_max = 2**zoom - 1

    min_tile_pt = Point(tile_x_min * tile_res, tile_y_min * tile_res)

    if not min_tile_pt.within(box(*etopo_file_data.total_bounds)):
        return

    etopo_idx = get_etopo_idx(etopo_file)
    px_contour_path = f"/data/misc/ETOPO_contours/contour_px_{etopo_idx}_EPSE3857_z{zoom}.pkl"
    if os.path.exists(px_contour_path):
        contour_px_df = pd.read_pickle(px_contour_path)
    else:
        etopo_px_x_min = tile_x_min * tile_res - px_buffer
        etopo_px_y_min = tile_y_min * tile_res - px_buffer
        etopo_px_x_max = (tile_x_max + 1) * tile_res + px_buffer
        etopo_px_y_max = (tile_y_max + 1) * tile_res + px_buffer

        etopo_pixel_bounds = np.array(
            [
                etopo_px_x_min,
                etopo_px_y_min,
                etopo_px_x_max,
                etopo_px_y_max,
            ]
        )

        etopo_x, etopo_y = etopo_file_data[["x", "y"]].values[0]

        etopo_px_coords = etopo_coords.copy()
        etopo_px_coords.geometry = etopo_px_coords.transform(
            lambda xy: wm_to_tile_coords(xy, zoom)
        )

        etopo_df_idx = etopo_px_coords[
            (etopo_px_coords.x >= etopo_x - 1)
            & (etopo_px_coords.x <= etopo_x + 1)
            & (etopo_px_coords.y >= etopo_y - 1)
            & (etopo_px_coords.y <= etopo_y + 1)
        ].index

        if etopo_x == 0:
            wrap_idx = etopo_px_coords.loc[
                (etopo_px_coords.x == 23)
                & (etopo_px_coords.y >= etopo_y - 1)
                & (etopo_px_coords.y <= etopo_y + 1)
            ].index

            etopo_df_idx = etopo_df_idx.union(wrap_idx)

        elif etopo_x == 23:
            wrap_idx = etopo_px_coords.loc[
                (etopo_px_coords.x == 0)
                & (etopo_px_coords.y >= etopo_y - 1)
                & (etopo_px_coords.y <= etopo_y + 1)
            ].index
            etopo_df_idx = etopo_df_idx.union(wrap_idx)

        etopo_px_coords = etopo_px_coords.loc[etopo_df_idx]

        etopo_dfs = []

        for _, row in etopo_px_coords.iterrows():
            etopo_df = etopo_px_contours(row.etopo_file, zoom)

            if etopo_x == 0 and row.x == 23:
                etopo_df.geometry = etopo_df.translate(
                    xoff=-(2**zoom) * tile_res
                ).make_valid(method="structure", keep_collapsed=False)
            elif etopo_x == 23 and row.x == 0:
                etopo_df.geometry = etopo_df.translate(
                    xoff=(2**zoom) * tile_res
                ).make_valid(method="structure", keep_collapsed=False)
            etopo_df.geometry = etopo_df.clip(
                etopo_pixel_bounds
                + [-2 * tile_res, -2 * tile_res, 2 * tile_res, 2 * tile_res],
                sort=True,
            ).make_valid(method="structure", keep_collapsed=False)

            if len(etopo_df) > 0:
                etopo_dfs.append(etopo_df)
        
        contour_px_df = pd.concat(etopo_dfs)

        contour_px_df = contour_px_df.sort_values(by=["layer_type", "level"])

        contour_px_df.to_pickle(px_contour_path)

    tile_coord_grid = np.meshgrid(
        np.arange(tile_x_min, tile_x_max + 1), np.arange(tile_y_min, tile_y_max + 1)
    )
    tile_coords = np.array(tile_coord_grid).T.reshape(-1, 2)

    tile_x_coords = tile_coords[:, 0]
    tile_y_coords = tile_coords[:, 1]

    with ThreadPoolExecutor() as executor:
        executor.map(
            write_onav_tiles,
            repeat(contour_px_df),
            tile_x_coords,
            tile_y_coords,
            repeat(zoom),
        )


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
                (z, x, y, blob_data),
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
                print(e)
                print(f"Unable to save tile ({x}, {y}, {z}) to {mbt_path}")
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
                (z, x, y, tile_data),
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
                print(e)
                print(f"Unable to save tile ({x}, {y}, {z}) to {mbt_path}")
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


sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)

if __name__ == "__main__":

    if not os.path.exists(ONAV_BASEMAP_DB):
        create_mbt_db(ONAV_BASEMAP_DB)

    if not os.path.exists(ONAV_LAND_MASK_DB):
        create_mbt_db(ONAV_LAND_MASK_DB)

    if not os.path.exists(ONAV_BATHY_DB):
        create_mbt_db(ONAV_BATHY_DB)

    etopo_files = sorted(
        glob.glob("/data/misc/ETOPO2022/15s/15s_surface_elev_netcdf/*.nc")
    )

    # etopo_files = [
    #     "/data/misc/ETOPO2022/15s/15s_surface_elev_netcdf/ETOPO_2022_v1_15s_N60W075_surface.nc"
    # ]

    # divide OSM files into managable chunks that match etopo bounds
    with ProcessPoolExecutor(max_workers=4) as executor:
        executor.map(split_osm_data, etopo_files)

    # extract contours from etopo and osm data
    print("Extract ETOPO contours.")
    with ProcessPoolExecutor(max_workers=6) as executor:
        executor.map(extract_etopo_contours, etopo_files)

    print("Extract pixel/zoom contours.")
    for etopo_file in etopo_files:
        with ProcessPoolExecutor(max_workers=6) as executor:
            executor.map(
                etopo_px_contours, repeat(etopo_file), reversed(range(1, 15))
            )

    for etopo_file in etopo_files:
        with ProcessPoolExecutor(max_workers=6) as executor:
            executor.map(process_etopo, repeat(etopo_file), reversed(range(1, 15)))
