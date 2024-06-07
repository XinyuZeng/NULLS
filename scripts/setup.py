from tqdm.notebook import tqdm_notebook
from tqdm import tqdm
import datetime
from typing import Dict, List
import json
import os
import numpy as npy
import pandas as pd
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
PROG_PATH = "./build/Release/NullsRevisited"
OUTPUT_PATH = os.path.abspath("./output.json")
EXP_REPEAT_TIMES = 10


def measure(null_rate: float, repeat_rate: float, range_max: int, try_count: int = 100, block_size: int = 1024, dist: str = "uniform", heuristic: str = ""):
    while True:
        os.system(
            f"taskset -c 1 {PROG_PATH} {null_rate} {repeat_rate} {range_max} {try_count} {block_size} {OUTPUT_PATH} {dist} {heuristic}")
        with open(OUTPUT_PATH, "r") as f:
            data = json.load(f)
        # Structure:
        # {
        #     "name": placeholder algorithm name,
        #     "try_count": try_count,
        #     "comp_time": compress time in seconds,
        #     "decomp_time": decompress time in seconds,
        #     "extra_sampling_time": extra sampling time in seconds,
        #     "spaced_to_dense_time": space to dense time in seconds,
        #     "size": compressed size in bytes
        # }
        # data is a list of such structures
        ok = True
        comp_times = [x["comp_time"] for x in data]
        if max(comp_times) < 30 * min(comp_times):
            return data
        print("Fluctuation too large, retrying...")
