# Data Preprocessing for Enefit's Kaggle dataset

This demonstrates the use of method chaining in processing multiple dataframes, and merge the processed dataframes into a final dataframe. These dataframes are then combined into a dataset for training Neural Network models and a dataset for training Gradient Boosted Decision Trees models.

To run this script, the dataset needs to be downloaded from [Kaggle](https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers) and unzipped to the directory of the script. Below is the expected result:

Two pickle files will be produced on success.

```
$python3 main.py
Data Readout Done.
Preprocessing begins.
100%|██████████████████████| 636/636 [01:13<00:00,  8.60it/s]
Neural Network data arrays done and saved:
NaN/Inf = 0, NonNum Cols = 0, shape = (634, 24, 5, 8, 14, 16)
NaN/Inf = 0, NonNum Cols = 0, shape = (634, 6, 5, 8, 14, 16)
NaN/Inf = 0, NonNum Cols = 0, shape = (634, 6, 8, 14, 7)
NaN/Inf = 0, NonNum Cols = 0, shape = (634, 24, 8, 14, 7)
NaN/Inf = 0, NonNum Cols = 0, shape = (634, 8, 14)
NaN/Inf = 0, NonNum Cols = 1, shape = (634, 1)
GBDT data arrays done and saved:
NaN/Inf = 7152, NonNum Cols = 0, shape = (2012496, 201)
```

The preprocessing functions are intentionally saved in multiple utilities scripts for clearity, and the scripts' names explain their purposes.

A `Data` class is defined in accordance with the competition's way of providing the data - each object expects a day-worth of data in several dataframes. While the current script has one chain of methods for each of the dataframes, it is also possible to have one chain for all of them, by having the `Data` class inheriting the `FocusableBase` defined but unused in `data_class.py`. `FocusableBase` lets us designate a dataframe-of-focus and redirect all method calls to it. `FocusableBase` was made for fun :) only.