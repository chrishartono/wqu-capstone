# wqu-capstone

## Overview

## Preparation
   1. Create virtual environment 
        > note: this experiment is conducted using Python version 3.12.x

        ```bash
        uv venv .venv --python=3.12
        ```

   2. Install required libraries
        ```bash
        # using native virtualenv
        pip install -r ./requirements.txt
        ```
        or 
        ```bash
        # using uv
        uv pip install -r ./requirements.txt
        ```

## Data
TODO

Dataset files can be found under `./dataset` directory.

Final (the most recent) dataset: `binance_1h_ohlcv_2023-2025.parquet`

```bash
# Obsolete dataset
mkdir data_local && unzip ./dataset/market_data.zip -d ./data_local
```

## Files
   - `backtester.py`: ...
   - `combinations.py`: ...
   - `comovement.py`: ...
   - TODO