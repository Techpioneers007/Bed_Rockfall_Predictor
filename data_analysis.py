import pandas as pd

df = pd.read_csv("Rockfall_Risk_Data.csv")
print(df.head())
print(df['rockfall_risk'].unique())

df['rockfall_risk'] = df['rockfall_risk'].map({'Low': 0, 'Medium': 1, 'High': 2})
df=df.drop("Unnamed: 0", axis = 1)
X = df.drop('rockfall_risk', axis=1)
y = df['rockfall_risk']


