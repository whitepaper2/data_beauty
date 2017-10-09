# Multidimensional scaling，简称MDS，中文翻译成多维尺度分析
# 其原理是利用成对样本间的相似性，去构建合适的低维空间，使得样本在此空间的距离和在高维空间中的样本间的相似性尽可能的保持一致。
# 我们可以用这种方式来可视化数据分布。
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import MDS

data = np.array([(5, 0, 0, 0, 0, 1, 0, 2, 1, 0),
                 (0, 0, 3, 0, 3, 0, 1, 0, 0, 1),
                 (2, 0, 0, 0, 0, 0, 1, 0, 0, 0),
                 (1, 0, 1, 0, 2, 0, 0, 0, 0, 1),
                 (5, 0, 2, 0, 0, 4, 2, 2, 3, 7),
                 (0, 3, 0, 1, 0, 0, 0, 0, 0, 0),
                 (0, 0, 0, 6, 0, 0, 0, 0, 0, 1),
                 (0, 5, 0, 0, 0, 0, 0, 0, 0, 0),
                 (0, 1, 0, 0, 0, 0, 0, 0, 0, 0),
                 (0, 2, 0, 0, 0, 0, 0, 0, 0, 0)
                 ]
                )
index = ['auto1', 'auto2', 'auto3', 'auto4', 'auto5', 'moto1', 'moto2', 'moto3', 'moto4', 'moto5']
columns = ['car', 'bike', 'cars', 'his', 'tires', 'she', 'ive', 'her', '#k', 'are']
Word = pd.DataFrame(data, index, columns)

mds = MDS()
mds.fit(data)
a = mds.embedding_
plt.scatter(a[0:5, 0], a[0:5, 1], color='turquoise')
plt.scatter(a[5:10, 0], a[5:10, 1], color='red')
plt.show()
