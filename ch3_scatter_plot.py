# data:lottery_1970.txt from http://lib.stat.cmu.edu/DASL/Datafiles/DraftLottery.html
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ds_arr = np.loadtxt("./lottery_1970.txt", skiprows=1, usecols=[3, 4])

# print(df)
# x:location,h:bandwidth,xp,yp:data points(vectors)
def loess(x, h, xp, yp):
    w = np.exp(-0.5 * (((x - xp) / h) ** 2) / np.sqrt(2 * np.pi * h ** 2))
    b = sum(w * xp) * sum(w * yp) - sum(w) * sum(w * xp * yp)
    b /= sum(w * xp) ** 2 - sum(w) * sum(w * xp ** 2)
    a = (sum(w * yp) - b * sum(w * xp)) / sum(w)
    return a + b * x


s1, s2 = [], []
for k in ds_arr[:, 0]:
    s1.append(loess(k, 5, ds_arr[:, 0], ds_arr[:, 1]))
    s2.append(loess(k, 100, ds_arr[:, 0], ds_arr[:, 1]))

# show scatter and loses plot
fig, ax = plt.subplots()
ax.scatter(ds_arr[:, 0], ds_arr[:, 1])
ax.plot(ds_arr[:, 0], s1, "k-", ds_arr[:, 0], s2, "g-")
ax.set_xlabel("Day of year")
ax.set_ylabel("Draft Number")
# ax.set_title("lottery_1970")
# ax.grid(True)
fig.tight_layout()
plt.show()
