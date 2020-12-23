import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestRegressor

seed = 7
np.random.seed(seed)

y_df = pd.read_csv('y.csv', index_col='ID')
y = y_df['pIC50'].values

x_df = pd.read_csv('rrelieff_descriptors.csv', index_col='ID')
x = x_df.values

x = StandardScaler().fit_transform(x)

x, x_holdout, y, y_holdout = train_test_split(x, y, test_size=0.333)

n_estimators = [2, 4, 8, 16, 32, 64, 128, 248, 512, 1024, 2048, 4096, 8192]
max_features = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
min_samples_split = [2, 4, 6]

parameters = dict(n_estimators=n_estimators, max_features=max_features, min_samples_split=min_samples_split)

model = RandomForestRegressor()
grid = GridSearchCV(model, parameters, scoring='neg_mean_squared_error', n_jobs=-1, cv=5, verbose=2)
grid_result = grid.fit(x, y)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
