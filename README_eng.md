> 🇰🇷 [한국어로 보기](./README.md)

# Recommender Systems (Modernized & Optimized)

This repository is a **modern and optimized reimplementation** of [moseskim/RecommenderSystems](https://github.com/moseskim/RecommenderSystems). The goal is to maintain the core educational content while updating the environment and optimizing the code for performance and compatibility with modern systems such as **macOS M-series** and **Linux AARCH64**.

## Key Updates

- Updated to **Python 3.10** or later versions
- Removed legacy or unsupported packages
- Improved compatibility across different system architectures (x86, ARM, etc.)
- Introduced modular data loading with:
  - **Pandas** (baseline)
  - **Dask** (for parallelized, out-of-core loading)
  - **PyArrow** (for columnar memory formats and faster I/O)

## Directory Structure

```
RecommenderSystems/
.
├── README.md
├── download_data.sh
├── notebooks
│   └── eda.ipynb
├── srcs
|   ├── __init__.py
|   ├── random_recommender.py
│   ├──  FM.py
...
└── utils.py
    ├── data_loader.py
    ├── metric_calculator.py
    └── models.py
```

## Environment

- **Python Version**: 3.10
- **Recommended Environment**: `venv` or `conda`
- Works across macOS (Intel & M-series), Linux (x86 & ARM)

## Required Packages

The following packages are required to run the project. These were selected for **performance, compatibility, and scalability**:

- `pandas`: for baseline data loading and processing
- `numpy`: numerical operations
- `matplotlib`: for plotting and visualization
- `dask`: scalable parallel computation and large dataset handling
- `pyarrow`: efficient in-memory columnar format and fast I/O

Install packages via pip:

```bash
pip install pandas numpy matplotlib dask pyarrow
```

⚠️ Note: You don't need to strictly follow specific package versions.
This project uses standard, widely-supported libraries. Feel free to install compatible versions based on your own Python environment. \
For reference, the author developed and tested this project on Apple M1 Pro, using Python 3.10 with Conda as the environment manager.


## Optimizations (In Progress)

This section tracks all enhancements and modernization tasks performed beyond the original repo.

### ✅ Data Loading

-  Replaced legacy CSV loading with `dask.dataframe` for large-scale handling
-  Added `pyarrow` loader for memory-efficient processing
-  Add benchmarking comparisons between loaders (coming soon)

### 🛠️ Model Performance
(coming soon)

### 📦 Packaging & CI
(coming soon)

## References

- Original Repo: [moseskim/RecommenderSystems](https://github.com/moseskim/RecommenderSystems)
