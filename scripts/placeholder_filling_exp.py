from setup import *


def var_null_rate_heuristics(null_rates: npy.ndarray, repeat_rate: float, range_max: int, try_count: int = 100, block_size: int = 1024, dist: str = "uniform", heuristics: List[str] = ["zero", "last"]):
    comp_times = {}
    decomp_times = {}
    sample_times = {}
    sizes = {}
    final_dfs = []
    for null_rate in tqdm(null_rates):
        datas = []
        for _ in range(EXP_REPEAT_TIMES):
            cur_data_list = []
            for heuristic in heuristics:
                cur_data_list.extend(measure(null_rate, repeat_rate, range_max,
                                     try_count, block_size, dist, heuristic))
                cur_data_list[-1]["name"] = heuristic
            datas.append(cur_data_list)
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
    final_df['dist'] = dist
    final_df['heuristic'] = final_df['name']
    return {"compress": comp_times, "decompress": decomp_times, "sample": sample_times}, sizes, final_df


null_rates = npy.linspace(0.0, 1.0, 40)
cur_range_max = 4096
cur_repeat_rate = 0.5
block_size = 1024 * 64
try_count = 16 * 8
dists = ["uniform", "gentle_zipf", "hotspot", "linear"]
heuristics = ["zero", "last", "linear", "random", "frequent", "smart"]


output_df = []
for dist in dists:
    comp_times, sizes, temp_df = var_null_rate_heuristics(
        null_rates, cur_repeat_rate, cur_range_max, dist=dist, try_count=try_count, block_size=block_size, heuristics=heuristics)
    output_df.append(temp_df)
output_df = pd.concat(output_df)
output_df.to_csv(f"./outputs/heuristics_{timestamp}.csv", index=False)
