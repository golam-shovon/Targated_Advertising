import pandas as pd
import numpy as np

data = {'Name': ['Billy', 'Sarah', 'Klara',  'Joseph', 'Bob', 'Sue'], 'Age': [23, 24, 26, 21, 15, 30], 'University': ['AIUB', 'AIUB' , 'DIU', 'AIUB' , 'NSU', 'UIU']}

df = pd.DataFrame(data)

p = df.as_matrix()

print(p)

