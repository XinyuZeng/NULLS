from setup import *


def count_to_color(rle_count: int, delta_count: int, bitpacking_count: int):
    total_count = rle_count + delta_count + bitpacking_count
    if total_count == 0:
        return (0, 0, 0,)
    return (rle_count / total_count, delta_count / total_count, bitpacking_count / total_count,)


def var_null_rate(null_rates: npy.ndarray, repeat_rate: float, range_max: int, try_count: int = 100, block_size: int = 1024, dist: str = "uniform"):
    comp_times = {}
    decomp_times = {}
    sample_times = {}
    sizes = {}
    colors = {}
    datas_null_rate = {}
    final_dfs = []

    for null_rate in tqdm(null_rates):
        # print(f"null_rate: {null_rate}")
        datas = []
        for _ in range(EXP_REPEAT_TIMES + 1):
            datas.append(measure(null_rate, repeat_rate, range_max,
                                 try_count, block_size, dist))
        dfs = []
        for i, d in enumerate(datas):
            df = pd.DataFrame(d)
            # add other col null_rate
            df["null_rate"] = null_rate
            df["exp_count"] = i
            dfs.append(df)
        final_dfs.append(pd.concat(dfs))
        datas = datas[1:]  # Discard the first run
        datas_null_rate[null_rate] = datas
        data = datas[0]
        for i in range(1, len(datas)):
            for j in range(len(data)):
                data[j]["comp_time"] += datas[i][j]["comp_time"]
                data[j]["decomp_time"] += datas[i][j]["decomp_time"]
                data[j]["extra_sampling_time"] += datas[i][j]["extra_sampling_time"]
                data[j]["size"] += datas[i][j]["size"]
                data[j]["rle_count"] += datas[i][j]["rle_count"]
                data[j]["delta_count"] += datas[i][j]["delta_count"]
                data[j]["bitpacking_count"] += datas[i][j]["bitpacking_count"]
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
                colors[name] = []
            comp_times[name].append(stat["comp_time"])
            decomp_times[name].append(stat["decomp_time"])
            sample_times[name].append(stat["extra_sampling_time"])
            sizes[name].append(stat["size"])
            colors[name].append(count_to_color(
                stat["rle_count"], stat["delta_count"], stat["bitpacking_count"]))
    final_df = pd.concat(final_dfs)
    final_df['dist'] = dist
    return {"compress": comp_times, "decompress": decomp_times, "sample": sample_times}, sizes, colors, datas_null_rate, final_df


null_rates = npy.linspace(0.0, 1.0, 40)
# null_rates = npy.linspace(0.49, 0.51, 100)
cur_range_max = 4000
cur_repeat_rate = 0.5
cur_try_count = 16 * 8
cur_block_size = 1024 * 64
dists = ["gentle_zipf", "hotspot", "linear", "uniform"]
output_df = []

for dist in dists:
    comp_times, sizes, colors, orig_data, temp_df = var_null_rate(
        null_rates, cur_repeat_rate, cur_range_max, dist=dist, try_count=cur_try_count, block_size=cur_block_size)
    output_df.append(temp_df)
output_df = pd.concat(output_df)
output_df.to_csv(f"./outputs/smartnulldense_{timestamp}.csv", index=False)
