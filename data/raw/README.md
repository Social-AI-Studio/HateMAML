This directory contains the raw pickle dataset files in the following format:

1. The file name should be of the format: `{dataset_name}_{split_name}.pkl`
2. Each pickle file should be a pickled Pandas [DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) of the following format:

| Column #  | Column Name | Non-Null Row Count | Dtype | Possible Values
|--- | ------ | -------------- | ----- | -----
| 0  | id     | `#row`  | int64 | integral
| 1  | text   | `#row`  | str or object  | string
| 2  | label  | `#row`  | int64 | integral
| 3  | lang   | `#row`  | str or object | `'en'`, `'hi'`, etc.

where `#row` is the number of rows in the DataFrame.