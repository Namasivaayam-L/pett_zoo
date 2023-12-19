import pandas as pd

data = pd.read_csv('gan/comb.csv',header=0)

data['0'] = data['0'].replace({r'[\[\]"]',''},regex=True)
print(type(data['0'].head(1)[0]))
data.to_csv('gan/formatted.csv', index=False)