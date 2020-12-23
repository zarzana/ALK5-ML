import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.ensemble import ExtraTreesRegressor

import numpy as np
import pandas as pd

import tensorflow as tf

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor

monitor = tf.keras.callbacks.EarlyStopping(
    monitor='loss',
    patience=100,
    min_delta=0.001,
    verbose=0,
    mode='auto',
    restore_best_weights=True,
    )

seed = 7
np.random.seed(seed)

path_y = 'y.csv'
y_df = pd.read_csv(path_y, index_col='ID')
y = y_df['pIC50'].values

path_x = 'rrelieff_descriptors.csv'
x_df = pd.read_csv(path_x, index_col='ID')
x = x_df.values

path_r = 'C:\Users\Zeki\OneDrive\IC_SOCORRO\stuff\python\sfs.csv'

x = StandardScaler().fit_transform(x)

def create_model():

    model = Sequential()

    layers = 16
        
    while layers > 0:

        layers -= 1

        model.add(Dense(256, kernel_initializer='he_uniform', activation='relu'))
        model.add(Dropout(0.1))

    model.add(Dense(1, kernel_initializer='he_uniform', activation='linear'))

    model.compile(loss='mse', optimizer='adam')
 
    return model

x, x_holdout, y, y_holdout = train_test_split(x, y, test_size=1/3)

regr = KerasRegressor(build_fn=create_model, verbose=0, epochs=10000, batch_size=32, callbacks=[monitor])

sfs = SFS(regr, k_features='best', forward=False, floating=True, scoring='neg_mean_squared_error', cv=5, n_jobs=-1, verbose=1)

sfs = sfs.fit(x, y, custom_feature_names=x_df.columns)

result_df = pd.DataFrame.from_dict(sfs.get_metric_dict()).T

result_df.to_csv(path_or_buf=path_r, sep=',', header=True, index=True, index_label=None, encoding='utf-8')
