# uses logistic regression with simple random split on the training_data.csv
# renders misclass rate between 0.25 and 0.40 depending on RNG



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as skl_lm
import sklearn.neighbors as skl_nb
from sklearn.preprocessing import StandardScaler

# load original training_data.csv
orig_training_data = pd.read_csv('Data/training_data.csv')
orig_cols = orig_training_data.columns.values
orig_N = len(orig_training_data)

# get dummies for time signature and for key
time_sign_df = orig_training_data['time_signature']
time_sign_dummies = pd.get_dummies(time_sign_df)
key_df = orig_training_data['key']
key_dummies = pd.get_dummies(key_df)

orig_training_data[['ts_1','ts_3','ts_4','ts_5']] = time_sign_dummies[[1,3,4,5]]

# orig_training_data[['key_0','key_1','key_2','key_3','key_4','key_5','key_6','key_7','key_8','key_9','key_10','key_11']] = \
#         key_dummies[list(range(0,12))]

orig_cols = orig_training_data.columns.values

# define training / testing sizes
N_TRAIN = 400
N_TEST = orig_N - N_TRAIN

# create training and testing data sets
np.random.seed(1)
idx = orig_training_data.index.isin(np.random.choice(orig_N, size=(N_TRAIN,), replace=False))
train = orig_training_data.iloc[idx]
test = orig_training_data.iloc[~idx]

# define input and output columns
cols_in = orig_cols[orig_cols != 'label']
# cols_in = cols_in[cols_in != 'duration']
# cols_in = cols_in[cols_in != 'tempo']
# cols_in = cols_in[cols_in != 'loudness']
# cols_in = cols_in[cols_in != 'valence']
# cols_in = cols_in[cols_in != 'energy']
# cols_in = cols_in[cols_in != 'acousticness']
# cols_in = cols_in[cols_in != 'mode']
cols_in = cols_in[cols_in != 'key']
cols_in = cols_in[cols_in != 'time_signature']

# cols_in = ['duration']
cols_out = ['label']


# define training and  testing inputs and outputs
X_train = train[cols_in].values

scaler_X = StandardScaler().fit(X_train)
X_train = scaler_X.transform(X_train)
y_train = train[cols_out].values.flatten()

X_test = test[cols_in].values
X_test = scaler_X.transform(X_test)
y_test = np.where(test[cols_out].values == 1, 'like', 'dislike').flatten()

# create and fit logisitic regression
log_reg = skl_lm.LogisticRegression(solver='lbfgs').fit(X_train, y_train)
knn = skl_nb.KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)

y_hat = np.where(log_reg.predict(X_test) == 1, 'like', 'dislike').flatten()
y_hat_knn = np.where(knn.predict(X_test) == 1, 'like', 'dislike').flatten()

# evaluate using crosstab
ct = pd.crosstab(y_test, y_hat, rownames=['true'], colnames=['prediction'])
print(ct)
print('\nMisclassified: ', np.sum(y_hat != y_test))
print('Misclass rate: ', np.mean(y_hat != y_test))
#
# ct = pd.crosstab(y_test, y_hat_knn, rownames=['true'], colnames=['prediction'])
# print(ct)
# print('\nMisclass rate: ', np.sum(y_hat_knn != y_test))
