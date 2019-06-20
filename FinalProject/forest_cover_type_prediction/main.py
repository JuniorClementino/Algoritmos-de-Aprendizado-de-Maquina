import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn import svm
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score
import time
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import *
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

pd.options.display.max_colwidth = 150
data = pd.read_csv("../input/train.csv")
test = pd.read_csv('../input/test.csv')
print(data.head())

# print(data.describe())

data_labels = data['Cover_Type'] - 1
data_train = data.drop(columns=['Cover_Type', 'Id'])
test_ids = test['Id']
data_test = test.drop(columns=['Id'])

# Normalizing the dataset: standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
scaler.fit(data_train)
scaler.fit(data_test)
data_train[:] = scaler.transform(data_train)
data_test[:] = scaler.transform(data_test)

models, results, tempos = list(), list(), list()

'''all_models = [XGBClassifier(n_estimators=256, objective='multi:softmax', num_class=7, max_depth=9,
                            colsample_bytree=.8, colsample_bylevel=.8, n_jobs=-1, random_state=2019),

              GradientBoostingClassifier(n_estimators=256, max_depth=9, random_state=2019),

              RandomForestClassifier(n_estimators=256, max_depth=9, random_state=2019, n_jobs=-1),

              MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(64, 32, 16), activation='logistic',
                            random_state=5)]

models.extend(all_models)'''


# Grid search for XGBClassifier:
xgboost = XGBClassifier(num_class=7)
parameters = [
    {'n_estimators': [128, 256, 400], 'max_depth': [4, 8, 16], 'max_features': ['log2', 'sqrt'],
     'colsample_bytree': [.2, .5, .9],
     'colsample_bylevel': [.2, .5, .9], 'colsample_bynode':[.2, .5, .9],
     'learning_rate': [0.1, 0.2, 0.04]}]

clf = GridSearchCV(xgboost, parameters, cv=5, verbose=1, n_jobs=-1)
clf.fit(data_train, data_labels)
print('XGBClassifier - Best estimator: ', clf.best_estimator_)
models.append(clf.best_estimator_)


# Grid search for RandomForestClassifier:
rfc = RandomForestClassifier()
parameters = [
    {'n_estimators': [100, 200, 400], 'max_depth': [5, 9, 13], 'max_features': ['log2', 'sqrt'],
     'min_samples_leaf':[1, 2, 4], 'criterion': ['gini', 'entropy'], 'min_samples_split':[2, 4, 8]}]
clf = GridSearchCV(rfc, parameters, cv=5, verbose=1, n_jobs=-1)
clf.fit(data_train, data_labels)
print('RandomForestClassifier - Best estimator: ', clf.best_estimator_)
models.append(clf.best_estimator_)


# Grid search for GradientBoostingClassifier:
gbc = GradientBoostingClassifier()
parameters = [
    {'n_estimators': [100, 200, 400], 'max_depth': [5, 9, 13], 'max_features': ['log2', 'sqrt'],
     'min_samples_leaf': [1, 2, 4, 8], 'loss':['deviance', 'exponential'],
     'learning_rate': [0.1, 0.05, 0.2], 'min_samples_split':[2, 4, 8]}]
clf = GridSearchCV(gbc, parameters, cv=5, verbose=1, n_jobs=-1)
clf.fit(data_train, data_labels)
print('GradientBoostingClassifier - Best estimator: ', clf.best_estimator_)
models.append(clf.best_estimator_)


kf = KFold(n_splits=5, shuffle=True, random_state=2019)
# Running models for the dataset:
# For each model:
for model in models:
    print('===========\n Model: ', model)
    # for each fold:
    accuracies = list()
    times = list()

    for i, (train_index, test_index) in enumerate(kf.split(data_train)):
        # Builds the train and validation dataset, according to the current fold:
        y_train, y_valid = data_labels.iloc[train_index].copy(), data_labels.iloc[test_index]
        X_train, X_valid = data_train.iloc[train_index, :].copy(), data_train.iloc[test_index, :].copy()
        start = time.time()
        model.fit(X_train, y_train)
        end = time.time()
        times.append(end - start)
        pred = model.predict(X_valid)
        acc = accuracy_score(pred, y_valid)
        accuracies.append(acc)

    print('Final results: \nMean accuracy:', np.mean(accuracies))
    print('Mean traning model time: ', np.mean(times))
    print('============')
    results.append(np.mean(accuracies))
    tempos.append(np.mean(times))

data = {'Accuracy': results, 'Time': tempos, 'Model': models}
dataframe = pd.DataFrame(data=data)
dataframe = dataframe.sort_values(by=['Accuracy'], ascending=False)

dataframe.to_csv('models_results.csv', index=False)

model_index = 0
for model in models:
    preds = model.predict(data_test) + 1
    dataframe = pd.DataFrame(data={'Id': test_ids, 'Cover_Type': preds})
    dataframe.to_csv('submission_model_' + str(model_index) + '.csv', index=False)
    model_index += 1
    print(dataframe.head())


'''
Grid Search for XGBoost:
Best Model: XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=0.7,
       colsample_bytree=0.9, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=14, min_child_weight=1, missing=None, n_estimators=200,
       n_jobs=1, nthread=None, num_class=7, objective='multi:softprob',
       random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
       seed=None, silent=True, subsample=1)'''
