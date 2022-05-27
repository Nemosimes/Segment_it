import pandas as pd
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from PreProcessing.Preprocessing import preprocessing, applyPCA
from helper_functions import write_to_csv, normalize_data, standardize_data, get_model_data

train_data, test_data = preprocessing(1)
x, y, ids, test_data = get_model_data(train_data, test_data)

''' UNCOMMENT FOR PCA
# pca = PCA(n_components=20)
# x = pca.fit_transform(x)
#
# explained_variance = pca.explained_variance_ratio_
# print(explained_variance)
'''

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)

clf = GradientBoostingClassifier(n_estimators=100, max_features="auto", random_state=23)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(accuracy_score(y_test, y_pred))


clf = GradientBoostingClassifier(n_estimators=100, max_features="auto", random_state=23)
clf.fit(x, y)
y_pred = clf.predict(test_data)
write_to_csv(ids, '../../predictions/predictedFromGB.csv', y_pred)
