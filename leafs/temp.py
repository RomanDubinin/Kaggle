# from sklearn.ensemble import RandomForestClassifier
# from numpy import genfromtxt, savetxt
# import numpy as np
# import csv

# #create the training & test sets, skipping the header row with [1:]
# dataset = genfromtxt('train.csv', delimiter=',', dtype=None)[1:]
# target = [x[1] for x in dataset]
# train = [x[2:] for x in dataset]
# test_id = genfromtxt('test.csv', delimiter=',', dtype='f8')[1:]
# test = [x[1:] for x in test_id]

# #create and train the random forest
# #multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
# rf = RandomForestClassifier(n_estimators=1000)
# rf.fit(train, target)

# probabilities = rf.predict_proba(test)

# trees = [x.decode('UTF-8') for x in rf.classes_]
# col_names =  ["Id"]  + trees

# ids = [int(test_id[i][0]) for i in range(len(probabilities))]


# print(type(ids[0]))


# result = [np.append([ids[i]], probabilities[i]) for i in range(len(probabilities))]


# print(type(result[1][0]))
# with open('submission2.csv', 'w', newline='') as f:
#     writer = csv.writer(f, delimiter=',')
#     writer.writerow(col_names)
#     for row in result:
#         writer.writerow([int(row[0]), *row[1:]])



import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import svm

from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

train = pd.read_csv('train.csv')
x_train = train.drop(['id', 'species'], axis=1).values
le = LabelEncoder().fit(train['species'])
y_train = le.transform(train['species']) #classes to numbers
scaler = StandardScaler().fit(x_train)
x_train = scaler.transform(x_train)

test = pd.read_csv('test.csv')
test_ids = test.pop('id')
x_test = test.values
scaler = StandardScaler().fit(x_test)
x_test = scaler.transform(x_test)

# rf = RandomForestClassifier(n_estimators=1000)
# rf.fit(x_train, y_train)
# y_test = rf.predict_proba(x_test)


params = {'C':[1, 10, 50, 100, 500, 1000, 2000], 
		  'tol': [0.001, 0.0001, 0.005],
		  'max_iter': np.arange(100, 500, 20),
		  'verbose': np.arange(0, 10)
		  }
log_reg = LogisticRegression(solver='lbfgs', multi_class='multinomial')
clf = GridSearchCV(log_reg, params, scoring='neg_log_loss', refit='True', n_jobs=1, cv=5)
clf.fit(x_train, y_train)
y_test = clf.predict_proba(x_test)
print(clf.best_params)

submission = pd.DataFrame(y_test, index=test_ids, columns=le.classes_)
submission.to_csv('submission.csv')