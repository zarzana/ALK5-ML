import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVR

seed = 7
np.random.seed(seed)

y_df = pd.read_csv('y.csv', index_col='ID')
y = y_df['pIC50'].values

x_df = pd.read_csv('rrelieff_descriptors.csv', index_col='ID')
x = x_df.values

x = StandardScaler().fit_transform(x)

x, x_holdout, y, y_holdout = train_test_split(x, y, test_size=1/3)

kernel = ['poly', 'rbf']
degree = [2, 3, 4, 5, 6]
C = [0.1, 0.5, 1, 2, 4, 6, 8, 10]
epsilon = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]

parameters = dict(kernel=kernel, degree=degree, C=C, epsilon=epsilon)

model = SVR()
grid = GridSearchCV(model, parameters, scoring='neg_mean_squared_error', n_jobs=-1, cv=5, verbose=2)
grid_result = grid.fit(x, y)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
