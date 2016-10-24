import pandas as pd
import numpy as np

from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, make_scorer, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt


def factorize(df):
    for column in df.columns:
        if (df[column].dtype != np.float64) and (df[column].dtype != np.int64):
            df[column] = df[column].factorize()[0]

data = pd.read_csv('train.csv', index_col='id')

x_data = data.drop("loss", axis=1)
y_data = data["loss"]

contFeatureslist = []
for colName,x in x_data.iloc[1,:].iteritems():
    if(not str(x).isalpha()):
        contFeatureslist.append(colName)

plt.figure(figsize=(13,9))
sns.boxplot(x_data[contFeatureslist])
plt.show()

for feature in contFeatureslist:
    x_data[feature] = (x_data[feature] - np.mean(x_data[feature])) / (max(x_data[feature]) - min(x_data[feature])) 

x_data[contFeatureslist] = normalize(x_data[contFeatureslist], axis=0)

factorize(x_data) # Сомнительное место, попробуй факторизовать по-другому

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.4, random_state=241)

###




grid = {'learning_rate': [0.6,0.7,0.8,0.9,1],
        'subsample': [0.8,0.9,1],
        'max_features': ['auto', 'sqrt', 'log2'],
        'n_estimators': [100, 500]}
clf = GradientBoostingRegressor(verbose=True)
gs = GridSearchCV(clf, grid, scoring='neg_mean_squared_error', cv=3, verbose=True)
gs.fit(x_data[:1000], y_data[:1000])

res = gs.cv_results_
df = pd.DataFrame.from_dict(res)
df.to_csv('gs_gbr_wit_norm.csv')

###
# plt.figure()
# original_params = {'n_estimators': 100, 'max_leaf_nodes': 4, 'random_state': 2,
#                    'min_samples_split': 5, 'verbose': True}
# for label, color, setting in [('learning_rate=0.1', 'turquoise',
#                                {'learning_rate': 0.1, 'subsample': 1.0}),
#                               ('learning_rate=0.1, subsample=0.5', 'gray',
#                                {'learning_rate': 0.1, 'subsample': 0.5}),
#                               ('learning_rate=0.1, max_features=2', 'magenta',
#                                {'learning_rate': 0.1, 'max_features': 2}),
#                               ("'learning_rate': 0.1, 'max_features': 5", 'orange',
#                                 {'learning_rate': 0.1, 'max_features': 'sqrt', 'subsample': 0.8})]:
#     params = dict(original_params)
#     params.update(setting)
#     clf = GradientBoostingRegressor(**params)
#     clf.fit(x_train, y_train)
#     # compute test set deviance
#     test_deviance = np.zeros((params['n_estimators'],), dtype=np.float64)
#     for i, y_pred in enumerate(clf.staged_decision_function(x_test)):
#         # clf.loss_ assumes that y_test[i] in {0, 1}
#         test_deviance[i] = clf.loss_(y_test, y_pred)
#     plt.plot((np.arange(test_deviance.shape[0]) + 1)[::5], test_deviance[::5],
#             '-', color=color, label=label)

# plt.legend(loc='upper left')
# plt.xlabel('Boosting Iterations')
# plt.ylabel('Test Set Deviance')

# plt.show()

# RandomForestRegressor:
# 1 0.0462952950736
# 2 0.305687200715
# 4 0.427534400224
# 8 0.48631077112
# 16 0.515588077484
# 32 0.530096760433
# 64 0.536901496607
# 128 0.541155974023
# 500 0.531931375445

# GradientBoostingRegressor:
# 1 0.0617750605578
# 8 0.28377584395
# 64 0.514547573643
# 500 0.562834613039
# 1000 0.565812314062
# 5000 0.558863904459




test = pd.read_csv('test.csv')
x_test = test.drop("id", axis=1)
factorize(x_test) # Сомнительное место, попробуй факторизовать по-другому


# for feature in contFeatureslist:
#     x_test[feature] = (x_test[feature] - np.mean(x_test[feature])) / (max(x_test[feature]) - min(x_test[feature])) 

x_test[contFeatureslist] = normalize(x_test[contFeatureslist], axis=0)

# with normalize: {'subsample': 1, 'max_features': 'auto', 'learning_rate': 0.6, 'n_estimators': 500} 
# 1" on full data : {'learning_rate': 0.6, 'n_estimators': 100, 'subsample': 1, 'max_features': 'auto'}
# 1: {'n_estimators': 500, 'max_features': 'log2', 'learning_rate': 0.6, 'subsample': 0.9}
# 90: {'n_estimators': 500, 'max_features': 'auto', 'learning_rate': 1, 'subsample': 0.8}

gbr = GradientBoostingRegressor(random_state=1, 
    n_estimators=500, 
    verbose=True, 
    learning_rate=0.6,
    max_features='auto',
    subsample=1)
gbr.fit(x_data, y_data)
y_gbr = gbr.predict(x_test)

submission = pd.DataFrame({'loss': y_gbr}, index=test["id"])
submission.to_csv('submission_gbr_norm_0.csv')