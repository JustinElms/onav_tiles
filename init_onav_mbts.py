import multiprocessing
import os
import sqlite3

import numpy as np

import utils
from build_tile_block import build_tile_block


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

    config = utils.read_config()

    mbt_path = config["OUTPUT_PATH"]

    if not os.path.exists(mbt_path):
        create_mbt_db(mbt_path)

    tile_indexes = utils.get_tile_indexes(8)

    inputs = np.column_stack((tile_indexes, np.repeat(mbt_path, len(tile_indexes))))

    with multiprocessing.Pool() as pool:
        pool.map(build_tile_block, inputs)
