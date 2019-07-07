import pandas as pd
import numpy as np
import random

df = pd.read_csv('adsratings.csv', index_col=False)

listsample = list(range(1, 68))

#df.loc[df['movieId'] > 100, 'movieId'] = random.choice(listsample)

for index_label, row_series in df.iterrows():

    if df.at[index_label, 'movieId'] >68:
        df.at[index_label, 'movieId'] = random.choice(listsample)
df.to_csv(r'adsratingsupdated.csv', index=False)

