# data from http://www.shanegalvin.com/analysing-dublin-marathon-data-1980-2016/
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# defaults
plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.rcParams.update({'font.size': 10})
plt.rcParams['xtick.major.pad'] = '5'
plt.rcParams['ytick.major.pad'] = '5'

plt.style.use('ggplot')
df = pd.read_excel("./dataset/lottery_1970.xlsx", sheetname="finishtime")


# print(df["Year"])


def loess(x, h, xp, yp):
    w = np.exp(-0.5 * (((x - xp) / h) ** 2) / np.sqrt(2 * np.pi * h ** 2))
    b = sum(w * xp) * sum(w * yp) - sum(w) * sum(w * xp * yp)
    b /= sum(w * xp) ** 2 - sum(w) * sum(w * xp ** 2)
    a = (sum(w * yp) - b * sum(w * xp)) / sum(w)
    return a + b * x


# calculate residual
s1_male, s2_male = [], []
s1_female, s2_female = [], []
for k in df["Year"]:
    s1_male.append(loess(k, 1, df["Year"], df["1st_Male2"]))
    s2_male.append(loess(k, 100, df["Year"], df["1st_Male2"]))
    s1_female.append(loess(k, 1, df["Year"], df["1st_Female2"]))
    s2_female.append(loess(k, 100, df["Year"], df["1st_Female2"]))
s1_male_residual = s1_male - df["1st_Male2"]
s2_male_residual = s2_male - df["1st_Male2"]
s1_female_residual = s1_female - df["1st_Male2"]
s2_female_residual = s2_female - df["1st_Male2"]

# show scatter and loses plot
fig, ax = plt.subplots(1, 3)


def plot_scatter_curve(ax, xaxis, yaxis1, yaxis2, xlabel, ylabel, width1=1, width2=100):
    ax.scatter(xaxis, yaxis1)
    ax.scatter(xaxis, yaxis2)
    yaxis1_s1, yaxis1_s2 = [], []
    yaxis2_s1, yaxis2_s2 = [], []
    for xi in xaxis:
        yaxis1_s1.append(loess(xi, width1, xaxis, yaxis1))
        yaxis1_s2.append(loess(xi, width2, xaxis, yaxis1))
        yaxis2_s1.append(loess(xi, width1, xaxis, yaxis2))
        yaxis2_s2.append(loess(xi, width2, xaxis, yaxis2))
    ax.plot(xaxis, yaxis1_s1, "k-", xaxis, yaxis1_s2, "k--")
    ax.plot(xaxis, yaxis2_s1, "g-", xaxis, yaxis2_s2, "g--")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


plot_scatter_curve(ax[0], df["Year"], df["1st_Male2"], df["1st_Female2"],
                   "Year", "Fininsh_time")
plot_scatter_curve(ax[1], df["Year"], s1_male_residual, s2_male_residual,
                   "Year", "Male_Residual")
plot_scatter_curve(ax[2], df["Year"], s1_female_residual, s2_female_residual,
                   "Year", "Female_Residual")
# ax.set_title("Finish time in Marathon between Male and Female")
# ax.grid(True)
fig.tight_layout()
plt.show()
