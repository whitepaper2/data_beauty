import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("./dataset/winequality-red.csv")
# print(df.head())
fig, ax = plt.subplots()
sns.pairplot(df)
plt.show()
