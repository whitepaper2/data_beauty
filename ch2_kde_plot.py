# %matplotlib inline
import matplotlib.pyplot as plt  # 导入
import pandas as pd
import seaborn as sns

sns.set(color_codes=True)  # 导入seaborn包设定颜色
fig, ax = plt.subplots(1, 2)

df = pd.read_excel("./dataset/president.xlsx")  # 导入数据集
# kernels = ["biw", "cos", "epa", "gau", "tri", "triw"]
# for k in kernels:
#     sns.kdeplot(df["Month"], kernel=k, label=k)
# sns.distplot(df["Month2"], bins=20, hist=True, rug=True)
plt.hist(df["Month2"], bins=5, cumulative=True, histtype="step")
plt.legend()
plt.tight_layout()
plt.show()
