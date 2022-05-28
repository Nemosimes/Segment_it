import pandas as pd
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
Anfrom sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from PreProcessing.Preprocessing import preprocessing, applyPCA
from helper_functions import write_to_csv, normalize_data, standardize_data, get_model_data
import lightgbm as lgb

train_data, test_data = preprocessing(2)
x, y, ids, test_data = get_model_data(train_data, test_data)
fvalue_Best = SelectKBest(f_classif, k=5)
X_kbest = fvalue_Best.fit_transform(x, y)
print(X_kbest)

print('Original number of features:', x.shape)
print('Reduced number of features:', x.shape)


# UNCOMMENT FOR PCA
# pca = PCA(n_components=20)
# x = pca.fit_transform(x)
#
# explained_variance = pca.explained_variance_ratio_
# print(explained_variance)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)

# def display(results):
#     print(f'Best parameters are: {results.best_params_}')
#     print("\n")
#     mean_score = results.cv_results_['mean_test_score']
#     std_score = results.cv_results_['std_test_score']
#     params = results.cv_results_['params']
#     for mean,std,params in zip(mean_score,std_score,params):
#         print(f'{round(mean,3)} + or -{round(std,3)} for the {params}')

# gbc = GradientBoostingClassifier()
# parameters = {
#     "n_estimators": [5, 50, 99,],
#     "max_depth": [2],
#     "learning_rate": [0.01, 0.05, 1,]
# }
# from sklearn.model_selection import GridSearchCV
#
# cv = GridSearchCV(gbc, parameters, cv=5)
# cv.fit(x_train, y_train)
# display(cv)


# clf = GradientBoostingClassifier(n_estimators=100, max_features="auto", random_state=23)
# learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1,]
# for learning_rate in learning_rates:
#     clf = GradientBoostingClassifier(n_estimators=99, learning_rate=learning_rate, max_features=21, max_depth=2,
#                                      random_state=23)
#     clf.fit(x_train, y_train)
#     y_pred = clf.predict(x_test)
#     print("Learning rate: ", learning_rate)
#     print("Accuracy score (training): {0:.3f}".format(clf.score(x_train, y_train)))
#     print("Accuracy score (validation): {0:.3f}".format(clf.score(x_test, y_test)))
#     print(accuracy_score(y_test, y_pred))

clf = GradientBoostingClassifier(n_estimators=99, learning_rate=0.05, max_features="auto", max_depth=2,
                                 random_state=23)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(accuracy_score(y_test, y_pred))

clf = GradientBoostingClassifier(n_estimators=99, learning_rate=0.05, max_features="auto", max_depth=2,
                                 random_state=23)
clf.fit(x, y)
y_pred = clf.predict(test_data)
write_to_csv(ids, '/Users/pierreehab/Desktop/Segment_it/predictions/predictedFromGBoosting.csv', y_pred)


# params = {}
# params['learning_rate'] = 0.1
# params['max_depth'] = 2
# params['n_estimators'] = 150
# # params['objective'] = 'multiclass'
# # params['boosting_type'] = 'gbdt'
# params['subsample'] = 1
# params['random_state'] = 23
# # params['colsample_bytree']=0.7
# # params['min_data_in_leaf'] = 55
# # params['reg_alpha'] = 1.7
# # params['reg_lambda'] = 1.11
#
# clf = lgb.LGBMClassifier(**params)
#
# clf.fit(x_train,y_train)
# y_pred = clf.predict(x_test)
# print(accuracy_score(y_test, y_pred))
#
# clf = lgb.LGBMClassifier(**params)
# clf.fit(x, y)
# y_pred = clf.predict(test_data)
# write_to_csv(ids, '/Users/pierreehab/Desktop/Segment_it/predictions/predictedFromGBoosting.csv', y_pred)