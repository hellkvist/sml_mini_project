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

# test data
test_data = pd.read_csv('Data/songs_to_classify.csv')

# define training / testing sizes
N_TRAIN = 750

# create training and testing data sets
# np.random.seed(1)
idx = orig_training_data.index.isin(np.random.choice(orig_N, size=(N_TRAIN,), replace=False))
train = orig_training_data.iloc[idx]
test = orig_training_data.iloc[~idx]

# define input and output columns
cols_in = orig_cols[orig_cols != 'label']
# cols_in = cols_in[cols_in != 'duration']
# cols_in = cols_in[cols_in != 'tempo']
# cols_in = cols_in[cols_in != 'loudness']
# cols_in = cols_in[cols_in != 'key']
# cols_in = cols_in[cols_in != 'time_signature']

# cols_in = ['duration']
cols_out = ['label']

# define training and  testing inputs and outputs
X_train = train[cols_in].values
scaler_X = StandardScaler().fit(X_train)
X_train = scaler_X.transform(X_train)
y_train = train[cols_out].values.flatten()

X_test = test_data[cols_in].values
X_test = scaler_X.transform(X_test)

# create and fit logisitic regression
log_reg = skl_lm.LogisticRegression(solver='lbfgs').fit(X_train, y_train)
knn = skl_nb.KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)

y_hat = log_reg.predict(X_test)
y_hat_df = pd.DataFrame(y_hat, columns=['label'])
test_data['label'] = pd.Series(y_hat, index=test_data.index)
print(test_data)
pd.plotting.scatter_matrix(test_data)
plt.show()