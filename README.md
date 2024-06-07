# NULLS

This repo contains code for the experiments in the DaMoN 2024 paper "NULLS! Revisiting Null Representation in Modern Columnar Formats"

## Setup

Tested on Debian 11. Intel(R) Xeon(R) Platinum 8375C CPU @ 2.90GHz

```bash
sudo apt-get install libpfm4-dev just build-essential cmake python3
```

<!-- Python scripts are tested on Python 3.9. The third-party dependencies are `numpy`, `pandas`, `matplotlib.pyplot`. -->
Python dependencies are in `requirements.txt`.

Code compiled in g++ 10.2.1

## Steps to reproduce the results

Clone the project recursively.

```bash
git clone --recursive git@github.com:XinyuZeng/NULLS.git
```

First build the target for later use:

```bash
just build
```

The cost of nulls:

```bash
just motivate
python3 scripts/motivate.py
```

C->P Conversion methods:

```bash
just bench_dense_to_spaced
```

Placeholder-filling strategies:

```bash
python3 scripts/placeholder_filling_exp.py
```

Compact vs. Placeholder w/o AVX512 encodings:

```bash
python3 scripts/compact_vs_placeholder.py
```

Compact vs. Placeholder on FLS w/ AVX512:

```bash
python3 scripts/fls.py
```

Compact vs. Placeholder on a vector primitive:

```bash
just bench_full
just bench_sv
```

C->P Conversion Miniblock size optimization:

```bash
just bench_miniblock_size
```

# Limitations

- Codecs involving FLS currently only support 1024 block size.
- Code with name prefix/suffix as "Dense" corresponds to "Compact" in the paper, and "Spaced" corresponds to "Placeholder" in the paper.