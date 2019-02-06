# uses logistic regression with simple random split on the training_data.csv
# renders misclass rate between 0.25 and 0.40 depending on RNG



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as skl_lm

# load original training_data.csv
orig_training_data = pd.read_csv('Data/training_data.csv')
orig_cols = orig_training_data.columns.values
orig_N = len(orig_training_data)

# define training / testing sizes
N_TRAIN = 350
N_TEST = orig_N - N_TRAIN

# create training and testing data sets
# np.random.seed(1)
idx = orig_training_data.index.isin(np.random.choice(orig_N, size=(N_TRAIN,), replace=False))
train = orig_training_data.iloc[idx]
test = orig_training_data.iloc[~idx]

# define input and output columns
cols_in = orig_cols[orig_cols != 'label']
cols_out = ['label']

# define training and  testing inputs and outputs
X_train = train[cols_in].values
y_train = train[cols_out].values.flatten()
X_test = test[cols_in].values
y_test = np.where(test[cols_out].values == 1, 'like', 'dislike').flatten()

# create and fit logisitic regression
log_reg = skl_lm.LogisticRegression(solver='lbfgs').fit(X_train, y_train)
y_hat = np.where(log_reg.predict(X_test) == 1, 'like', 'dislike').flatten()

# evaluate using crosstab
ct = pd.crosstab(y_test, y_hat, rownames=['true'], colnames=['prediction'])
print(ct)
print('\nMisclass rate: ', np.mean(y_hat != y_test))