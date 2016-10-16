import pandas as pd
import numpy as np

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, make_scorer, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

def factorize(df):
    for column in df.columns:
        if (df[column].dtype != np.float64) and (df[column].dtype != np.int64):
            df[column] = df[column].factorize()[0]

def NaN_to_mean(train_matrix):
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp = imp.fit(train_matrix)
    return imp.transform(train_matrix)



train = pd.read_csv('train.csv')
factorize(train)
x_train = train.drop(['Id', 'SalePrice'], axis=1).values

x_train = NaN_to_mean(x_train)

y_train = train['SalePrice']

test = pd.read_csv('test.csv')
factorize(test)
test_ids = test.pop('Id')
x_test = test.values
x_test = NaN_to_mean(x_test)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

### grid search to find optimal params

num_of_splits = 3
grid = {'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1],
        'subsample': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7,0.8,0.9,1],
        'max_features': ['auto', 'sqrt', 'log2'],
        'n_estimators': [100, 200, 500, 800]}
clf = GradientBoostingRegressor(verbose=True)
gs = GridSearchCV(clf, grid, scoring='neg_mean_squared_error', cv=num_of_splits, verbose=True)
gs.fit(x_train_scaled, y_train)

res = gs.cv_results_
df = pd.DataFrame.from_dict(res)
df.to_csv('gs_gbr.csv')

# {'subsample': 0.6, 'max_features': 'auto', 'n_estimators': 500, 'learning_rate': 0.1}
###


# rf = RandomForestRegressor(random_state=1, n_estimators=10000)
# rf.fit(x_train_scaled, y_train)


y_test = gs.predict(x_test_scaled)
y_test[np.nonzero(y_test<0)] = 0
submission = pd.DataFrame({'SalePrice': y_test}, index=test_ids)
submission.to_csv('submission.csv')

