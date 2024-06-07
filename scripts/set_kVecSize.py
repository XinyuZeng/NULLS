import re
import sys
from pathlib import Path


def replace(filename, pattern, replacement):
    f = open(filename)
    s = f.read()
    f.close()
    if re.search(pattern, s).group(0) == replacement:
        return
    s = re.sub(pattern, replacement, s)
    f = open(filename, 'w')
    f.write(s)
    f.close()


if __name__ == "__main__":
    kVecSize = sys.argv[1]
    pattern = r"static constexpr int kVecSize = .*"
    replacement = "static constexpr int kVecSize = " + str(kVecSize) + ";"
    repo_path = Path(__file__).resolve().parents[1]
    replace("{}/mem/utils/macros.hpp".format(repo_path), pattern, replacement)
