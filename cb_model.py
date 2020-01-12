# Model for making predictions

# Load libraries
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool, cv

# Load data
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')

# Model Definition
cb_model = CatBoostClassifier(iterations=100,
                              depth=3,
                              learning_rate=0.1,
                              loss_function='MultiClass')

# Fit model
# index of ORIGIN col; only categorical
cat_ft_indices = np.where(X_train.dtypes != np.int64)[0]

cb_model.fit(X_train,
             y_train,
             cat_features=cat_ft_indices,
             eval_set=(X_test, y_test),
             plot=True)

# Accuracy & Cross-Validation
cb_accuracy = cb_model.score(X_train, y_train)

# cv
train_pool = Pool(X_train,
                  y_train,
                  cat_ft_indices)

cross_val_paramt = cb_model.get_params()

cross_val_results = cv(pool=train_pool,
                       params=cross_val_paramt,
                       fold_count=10,
                       plot=True)

cb_cross_val_acc_avg = np.mean(cross_val_results['test-MultiClass-mean'])

cb_cross_val_acc_min = np.min(cross_val_results['test-MultiClass-mean'])

cb_cross_val_acc_max = np.max(cross_val_results['test-MultiClass-mean'])

print('Average Cross Validation Score:',
      round(cb_cross_val_acc_avg*100, 2),
      '\n'
      'Maximum Cross Validation Score:',
      round(cb_cross_val_acc_max*100, 2),
      '\n',
      'Minimum Cross Validation Score:',
      round(cb_cross_val_acc_min*100, 2))