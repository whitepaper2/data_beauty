# chapter two: workshop in numpy
# five methods to create vector
import numpy as np

vec1 = np.array([0., 1., 2., 3., 4.])
vec2 = np.arange(0, 5, 1, dtype=float)
vec3 = np.linspace(0, 4, 5)
vec4 = np.zeros(5)
for i in range(5):
    vec4[i] = i
# vec5 = np.loadtxt("./dataset/president.xlsx")
print("create array:", vec1, vec2, vec3, vec4, sep="\n")
# basic operation
v1 = vec1 + vec2
vec1 += vec2
v3 = 2 * vec3
v4 = vec4 + 3
v5 = np.sin(vec4)
print("array operation:", v1, vec1, v3, v4, v5.tolist())

from numpy import *


def kde(z, w, xv):
    return sum(np.exp(-0.5 * ((z - xv) / w) ** 2) / np.sqrt(2 * np.pi * w ** 2))


d = loadtxt("./dataset/president.txt", usecols=(3,))
w = 2.5
for x in linspace(min(d) - w, min(d) + w, 1000):
    print(x, kde(x, w, d))
# shape()\reshape()\slice()
d = np.linspace(0, 11, 12)
d.shape = (3, 4)
print("slice:", d, d[0, :], d[:, 1], d[0, 1], d[0:2, 1], d[0:1, 1:2], d[:, [2, 0]])

k = np.array([False, True, True])
print(d[k, :])
