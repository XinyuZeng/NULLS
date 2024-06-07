from setup import *


def var_null_rate(null_rates: npy.ndarray, repeat_rate: float, range_max: int, try_count: int = 100, block_size: int = 1024, dist: str = "uniform"):
    comp_times = {}
    decomp_times = {}
    sample_times = {}
    sizes = {}
    final_dfs = []
    for null_rate in tqdm(null_rates):
        datas = []
        for _ in range(EXP_REPEAT_TIMES):
            datas.append(measure(null_rate, repeat_rate, range_max,
                                 try_count, block_size, dist))
        # assert EXP_REPEAT_TIMES > 2 # discard first run
        dfs = []
        for i, d in enumerate(datas):
            df = pd.DataFrame(d)
            # add other col null_rate
            df["null_rate"] = null_rate
            df["exp_count"] = i
            dfs.append(df)
        final_dfs.append(pd.concat(dfs))
        data = datas[0]
        for i in range(1, len(datas)):
            for j in range(len(data)):
                data[j]["comp_time"] += datas[i][j]["comp_time"]
                data[j]["decomp_time"] += datas[i][j]["decomp_time"]
                data[j]["extra_sampling_time"] += datas[i][j]["extra_sampling_time"]
                data[j]["size"] += datas[i][j]["size"]
        for j in range(len(data)):
            data[j]["comp_time"] /= len(datas)
            data[j]["decomp_time"] /= len(datas)
            data[j]["extra_sampling_time"] /= len(datas)
            data[j]["size"] /= len(datas)

        for stat in data:
            name = stat["name"]
            if name not in comp_times:
                comp_times[name] = []
                decomp_times[name] = []
                sample_times[name] = []
                sizes[name] = []
            comp_times[name].append(stat["comp_time"])
            decomp_times[name].append(stat["decomp_time"])
            sample_times[name].append(stat["extra_sampling_time"])
            sizes[name].append(stat["size"])
    final_df = pd.concat(final_dfs)
    final_df.to_csv(f"./outputs/fls_{dist}_{timestamp}.csv", index=False)
    return {"compress": comp_times, "decompress": decomp_times, "sample": sample_times}, sizes


null_rates = npy.linspace(0.0, 0.98, 25)
cur_range_max = 4096
cur_repeat_rate = 0.5
try_count = 1024
block_size = 1024
dists = ["gentle_zipf", "linear"]

for dist in dists:
    comp_times, sizes = var_null_rate(
        null_rates, cur_repeat_rate, cur_range_max, dist=dist, try_count=try_count, block_size=block_size)
