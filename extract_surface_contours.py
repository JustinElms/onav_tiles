import os

import cartopy.crs as ccrs
import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely import box, Polygon
from scipy.ndimage import binary_fill_holes


levels = np.array([100, 200, 500, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 15000])


def get_contour_df(
    surface: np.array,
    levels: np.array,
    contour_type: str,
    land_geom,
    x_coords,
    y_coords,
) -> gpd.GeoDataFrame:
    contour_df = gpd.GeoDataFrame(
        columns=["geometry", "level", "layer_type"], geometry="geometry", crs=3857
    )
    for idx, level in enumerate(levels):
        mask = surface.z.values.copy()
        if contour_type == "bathy":
            mask = -1 * mask
        mask[mask < level] = 0
        mask[np.isnan(mask)] = 0
        mask[mask >= level] = 1
        mask = binary_fill_holes(mask)

        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        level_contours = []
        for contour in contours:
            if contour.size > 4:
                new_geom = Polygon(
                    np.vstack(
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
                )
                level_contours.append(new_geom)
        n_contours = len(level_contours)
        if n_contours > 0:
            level_df = gpd.GeoDataFrame(
                columns=["geometry", "level", "layer_type"], geometry="geometry", crs=3857
            )
            level_df.geometry = level_contours
            level_df.level = idx + 1
            level_df.layer_type = contour_type
            contour_df = pd.concat([contour_df, level_df])

    contour_df.geometry = contour_df.make_valid(
        method="structure", keep_collapsed=False
    )
    contour_df = contour_df[~contour_df.is_empty]
    contour_df = contour_df.dissolve(by=["level"], as_index=False)

    if contour_type != "bathy":
        contour_df.geometry = contour_df.intersection(land_geom)

    return contour_df


def clip_df_contours(row, contours):
    clipped = gpd.clip(contours, row.ext)
    clipped["x"] = row.x
    clipped["y"] = row.y
    return clipped


def get_etopo_paths(pc_ext) -> list:

    x0, y0, x1, y1 = pc_ext
    pc_bbox = box(*pc_ext)

    cx, cy = pc_bbox.centroid.xy
    cx = cx[0]
    cy = cy[0]

    xx = "E"
    xxx = int(15 * np.floor(abs(cx) / 15))
    if cx < 0:
        xx = "W"
        xxx = int(15 * np.ceil(abs(cx) / 15))

    yy = "N"
    yyy = int(15 * np.ceil(cy / 15))
    if cy < -15:
        yy = "S"

    bed_path = f"/data/misc/ETOPO2022/15s/15s_bed_elev_netcdf/ETOPO_2022_v1_15s_{yy}{yyy:02d}{xx}{xxx:03d}_bed.nc"
    surf_path = f"/data/misc/ETOPO2022/15s/15s_surface_elev_netcdf/ETOPO_2022_v1_15s_{yy}{yyy:02d}{xx}{xxx:03d}_surface.nc"

    return [bed_path, surf_path]


def extract_surface_contours(mbt_df, land_df, pc_ext) -> gpd.GeoDataFrame:

    bed_path, surf_path = get_etopo_paths(pc_ext)

    x0, y0, x1, y1 = pc_ext

    bed = None
    surface = xr.open_dataset(surf_path).sel(
        lon=slice(x0 - 1, x1 + 1), lat=slice(y1 - 1, y0 + 1)
    )
    if os.path.exists(bed_path):
        bed = xr.open_dataset(bed_path).sel(
            lon=slice(x0 - 1, x1 + 1), lat=slice(y1 - 1, y0 + 1)
        )
        ice = surface.copy()

        tmp_surface = surface.z.values.copy()
        tmp_surface[tmp_surface < 0] = np.nan
        tmp_surface[bed.z.values > tmp_surface] = np.nan

        diff = tmp_surface - bed.z.values

        ice["z"].data = ice["z"].data * np.nan
        ice["z"].data[diff >= 10] = surface["z"].data[diff >= 10]

        if len(diff[diff < 10]) == 0:
            surface["z"].data = surface["z"].data * np.nan
        else:
            surface["z"].data[diff >= 10] = np.nanmax(surface["z"].data[diff < 10])

    # convert lat lon to tile pixel coordinates
    pc_crs = ccrs.PlateCarree()
    wm_crs = ccrs.epsg("3857")
    wm_proj_bounds = [-20037508.34, -20048966.1, 20037508.34, 20048966.1]

    lon_grid, lat_grid = np.meshgrid(surface.lon.values, surface.lat.values)
    wm_coords = wm_crs.transform_points(pc_crs, lon_grid, lat_grid)[:, :, :2]
    wm_x_coords = wm_coords[:, :, 0]
    wm_y_coords = wm_coords[:, :, 1]

    x_coords = (
        4096
        * (2**14)
        * (wm_x_coords - wm_proj_bounds[0])
        / (wm_proj_bounds[2] - wm_proj_bounds[0])
    )
    y_coords = (
        4096
        * (2**14)
        * (wm_y_coords - wm_proj_bounds[1])
        / (wm_proj_bounds[3] - wm_proj_bounds[1])
    )

    x_coords = np.round(x_coords).astype(int)
    y_coords = np.round(y_coords).astype(int)

    land_geom = land_df.union_all()

    land_contours = get_contour_df(
        surface, levels, "land", land_geom, x_coords, y_coords
    )
    land_contours = pd.concat([land_df, land_contours])
    bathy_contours = get_contour_df(
        surface, levels, "bathy", land_geom, x_coords, y_coords
    )

    output_df = pd.concat([bathy_contours, land_contours])
    if bed:
        ice_contours = get_contour_df(
            ice, np.insert(levels, 0, -100), "ice", land_geom, x_coords, y_coords
        )
        output_df = pd.concat([output_df, ice_contours])

    output_df = mbt_df.apply(lambda row: clip_df_contours(row, output_df), axis=1)
    output_df = pd.concat(output_df.values)
    output_df.reset_index(drop=True, inplace=True)

    output_df.sort_values(by=["x", "y", "level"], ignore_index=True, inplace=True)

    dup_geoms = output_df.normalize().drop_duplicates(keep="last")

    output_df = output_df.loc[dup_geoms.index].reset_index(drop=True)

    return output_df
