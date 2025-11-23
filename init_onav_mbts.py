import multiprocessing
import os
import sqlite3

import numpy as np

from build_tile_block import build_tile_block


def get_tile_indexes(zoom) -> list:
    xyz = np.array(
        np.meshgrid(range(2**zoom), range(2**zoom), [zoom], indexing="ij")
    ).T.reshape(-1, 3)

    return xyz.astype(int)


def create_mbt_db(name: str) -> None:

    con = sqlite3.connect(name)
    cur = con.cursor()
    init_statements = [
        "CREATE TABLE metadata (name text, value text);",
        "CREATE TABLE tiles (zoom_level integer, tile_column integer, tile_row integer, tile_data blob);",
        "CREATE UNIQUE INDEX name on metadata (name);",
        "CREATE UNIQUE INDEX tile_index on tiles (zoom_level, tile_column, tile_row);",
    ]
    for stmt in init_statements:
        cur.execute(stmt)
    con.close()


if __name__ == "__main__":

    mbt_path = "/data/misc/shapes/onav.mbtiles"

    if not os.path.exists(mbt_path):
        create_mbt_db(mbt_path)

    tile_indexes = get_tile_indexes(8)

    inputs = np.column_stack((tile_indexes, np.repeat(mbt_path, len(tile_indexes))))

    for input in inputs:
        build_tile_block(input)

    with multiprocessing.Pool() as pool:
        pool.map(build_tile_block, inputs)
