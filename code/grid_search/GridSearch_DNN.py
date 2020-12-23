import numpy as np
import pandas as pd

import tensorflow as tf

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

tf.compat.v1.disable_eager_execution()

seed = 7
np.random.seed(seed)

monitor = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    patience=100,
    min_delta=0.001,
    verbose=0,
    mode='auto',
    restore_best_weights=True,
    )

def create_model(dropout_rate=0.0, neurons=32, layers=1):

    model = Sequential()
        
    while layers > 0:

        layers -= 1

        model.add(Dense(neurons, kernel_initializer='he_uniform', activation='relu'))
        model.add(Dropout(dropout_rate))

    model.add(Dense(1, kernel_initializer='he_uniform', activation='linear'))

    model.compile(loss='mse',
                  optimizer='adam')
 
    return model

x_df = pd.read_csv('rrelieff_descriptors.csv', index_col='ID')
y_df = pd.read_csv('y.csv', index_col='ID')

x = x_df.values
y = y_df['pIC50'].values

x = StandardScaler().fit_transform(x)

x, x_holdout, y, y_holdout = train_test_split(x, y, test_size=1/3)

model = KerasRegressor(build_fn=create_model, verbose=0)

dropout_rate = [0.0, 0.1, 0.2, 0.3]
neurons = [64, 128, 256, 512]
layers = [8, 16, 32]

param_grid = dict(dropout_rate=dropout_rate, neurons=neurons, layers=layers)

grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_squared_error', n_jobs=-1, cv=5, verbose=2)
grid_result = grid.fit(x, y, callbacks=[monitor], batch_size=32, epochs=10000)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
