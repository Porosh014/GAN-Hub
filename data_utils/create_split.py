import os
import numpy as np
import pandas as pd

def create_train_val_split(dataset_root):
    split_path = os.path.join(dataset_root, 'list_eval_partition.csv')

    partition = pd.read_csv(split_path)

    train_partition = partition.loc[partition['partition'] == 0]
    val_partition = partition.loc[partition['partition'] == 1]
    test_partition = partition.loc[partition['partition'] == 2]

    train_splits = list(train_partition.iloc[:, 0])
    val_splits = list(val_partition.iloc[:, 0])
    test_splits = list(test_partition.iloc[:, 0])

    return train_splits, val_splits, test_splits
