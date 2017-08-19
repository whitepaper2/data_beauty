# data from http://www.shanegalvin.com/analysing-dublin-marathon-data-1980-2016/
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_excel("./lottery_1970.xlsx", sheetname="finishtime")
# print(df["Year"])


def loess(x, h, xp, yp):
    w = np.exp(-0.5 * (((x - xp) / h) ** 2) / np.sqrt(2 * np.pi * h ** 2))
    b = sum(w * xp) * sum(w * yp) - sum(w) * sum(w * xp * yp)
    b /= sum(w * xp) ** 2 - sum(w) * sum(w * xp ** 2)
    a = (sum(w * yp) - b * sum(w * xp)) / sum(w)
    return a + b * x


s1_male, s2_male = [], []
s1_female, s2_female = [], []
for k in df["Year"]:
    s1_male.append(loess(k, 1, df["Year"], df["1st_Male2"]))
    s2_male.append(loess(k, 100, df["Year"], df["1st_Male2"]))
    s1_female.append(loess(k, 1, df["Year"], df["1st_Female2"]))
    s2_female.append(loess(k, 100, df["Year"], df["1st_Female2"]))

# show scatter and loses plot
fig, ax = plt.subplots()
ax.scatter(df["Year"], df["1st_Male2"])
ax.scatter(df["Year"], df["1st_Female2"])
ax.plot(df["Year"], s1_male, "k-", df["Year"], s2_male, "k--")
ax.plot(df["Year"], s1_female, "g-", df["Year"], s2_female, "g--")
ax.set_xlabel("Year")
ax.set_ylabel("Fininsh_time")
# ax.set_title("Finish time in Marathon between Male and Female")
# ax.grid(True)
fig.tight_layout()
plt.show()
