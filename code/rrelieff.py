import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

np.seterr(divide='ignore')


class RReliefF:

    def __init__(self, k=10, sigma=30):
        
        self.k = k
        self.sigma = sigma

        self.weights = None
        self.w_track = None

        distance_list = [np.exp(-((n + 1) / sigma) ** 2) for n in range(k)]
        self._distance = distance_list / np.sum(distance_list)

    def fit(self, x, y):

        m = x.shape[0]
        
        self.weights = np.zeros([x.shape[1],1])
        self.w_track = np.zeros([m, x.shape[1]])

        n_dc = 0
        n_da = np.zeros([x.shape[1],1])
        n_dcandda= np.zeros([x.shape[1],1])
        y_range = np.max(y) - np.min(y)

        for i in tqdm(range(m)):

            b = x[i, :]
            difference = (x - b)**2
            sum_difference = np.sum(difference, axis = 1)
            neighbour_index = np.argsort(sum_difference)
            neighbours = x[neighbour_index][1:]
            knn = neighbours[:self.k]

            x_knn, neighbour_index = knn, neighbour_index[1:]
            y_knn = y[neighbour_index]

            x_i = x[i, :]
            y_i = y[i]

            for j in range(self.k):

                n_dc += (np.abs(y_i-y_knn[j])/y_range) * self._distance[j]

                for a in range(x.shape[1]):

                    diff = np.abs(x_i[a] - x_knn[j][a]) / np.max(x[:, a]) - np.min(x[:, a])

                    n_da[a] = n_da[a] + self._distance[j] * diff

                    n_dcandda[a] = n_dcandda[a] + (np.abs(y_i-y_knn[j])/y_range) * self._distance[j] * diff

            for a in range(x.shape[1]):
                self.w_track[i, a] = n_dcandda[a] / n_dc - ((n_da[a] - n_dcandda[a]) / (m - n_dc))

        for a in range(x.shape[1]):
            self.weights[a] = n_dcandda[a]/n_dc - ((n_da[a]-n_dcandda[a])/(m-n_dc))


x = StandardScaler().fit_transform(pd.read_csv('unique_descriptors.csv', index_col='ID').values)
y = pd.read_csv('y.csv', index_col='ID').values

method = RReliefF()
method.fit(x, y)

data = dict(Descriptor=list(pd.read_csv('unique_descriptors.csv', index_col='ID').columns),
            Values=list(np.ravel(method.weights)))

df = pd.DataFrame.from_dict(data)
df.to_csv('rrelieff.csv', index=False)
