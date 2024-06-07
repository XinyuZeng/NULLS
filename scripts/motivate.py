import os
import time
import csv

# executable = "/home/xinyu/arrow-rs/target/release/parquet-read"
executable = "./build/Release/read_pq"

null_percent = [0, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 97, 99, 100]
repetition_cnt = 5

result = []

for r in range(repetition_cnt):
    for n in null_percent:
        os.system("sudo sync; echo 3 | sudo tee /proc/sys/vm/drop_caches")
        t = os.popen(executable + " " + str(n) + ".parquet").readlines()[0]
        t_hot = os.popen(executable + " " + str(n) + ".parquet").readlines()[0]
        result.append(
            {"null_percent": n, "time_cold": float(t), "time_hot": float(t_hot), "repetition": r})

with open("outputs/motivate_dict.csv", "w") as f:
    # with open("outputs/motivate_dict_rs.csv", "w") as f:
    writer = csv.DictWriter(
        f, fieldnames=["null_percent", "time_cold", "time_hot", "repetition"])
    writer.writeheader()
    writer.writerows(result)
