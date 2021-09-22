# Multilingual Hate Speech

## Accessing the datasets
To process the datasets:

1. place the raw pickled Pandas [DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) at directory `data/raw/`. This DataFrame is expected to be in the format as described in `data/raw/README.md`.
2. Execute `process_data.py` to convert raw data to processed data.
3. Load this processed data in your model script as described in the *Developing baselines* section.

## Developing baselines
For example baseline code, refer to `baselines/example.py`

It should be executed from this root project directory:

```
python3 baselines/example.py
```
