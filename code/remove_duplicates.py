import pandas as pd
import numpy as np
from itertools import combinations
from tqdm import tqdm
from sklearn.metrics import r2_score

x_df = pd.read_csv('calculated_descriptors.csv', index_col='ID')

pairs = list(combinations(x_df.columns, 2))

remove = []

for pair in tqdm(pairs):

    a0 = x_df[pair[0]].values
    a1 = x_df[pair[1]].values

    if r2_score(a1, a0) > 0.95:
        remove.append(pair[0])

remove = list(dict.fromkeys(remove))

x_df.drop(columns=remove, inplace=True)

x_df.to_csv('unique_descriptors.csv')
