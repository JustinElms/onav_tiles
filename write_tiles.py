import sqlite3

import geopandas as gpd
import mapbox_vector_tile


def write_tiles(mbt_df: gpd.GeoDataFrame, mbt_path: str) -> None:

    xy_vals = mbt_df[["x", "y"]].drop_duplicates().values

    tile_data = []
    for x, y in xy_vals:
        tile = mbt_df.loc[(mbt_df.x == x) & (mbt_df.y == y)].reset_index(drop=True)
        features = []
        for idx, layer in tile.iterrows():
            feat = {
                "geometry": layer.geometry,
                "properties": {
                    "id": layer.index,
                    "layer": f"{layer.type}_{layer.level}",
                },
            }
            features.append(feat)
        tile_data.append(
            [
                x,
                y,
                14,
                mapbox_vector_tile.encode([{"name": "output", "features": features}]),
            ]
        )

    con = sqlite3.connect("test.mbtiles")
    cur = con.cursor()

    for tile in tile_data:
        try:
            cur.execute(
                "INSERT INTO tiles VALUES (?, ?, ?, ?)",
                (tile[0], tile[1], tile[2], tile[3]),
            )
        except Exception as e:
            print(e)
            print(f"Unable to save tile {[*tile[:3]]}")

    con.commit()
    con.close()
