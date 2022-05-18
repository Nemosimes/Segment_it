import inline
import matplotlib
import pandas as pd
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from PreProcessing.Preprocessing import preprocessing
from helper_functions import write_to_csv, normalize_data, standardize_data, get_model_data

train_data, test_data = preprocessing(1)
x, y, ids, test_data = get_model_data(train_data, test_data)

x_train, x_test, y_train, y_test,  = train_test_split(x, y, test_size=0.2, random_state=23)

# create a random forest classifier
rfc = RandomForestClassifier(n_estimators=100, max_features="auto", random_state=23)

# train the model using the split training set
rfc.fit(x_train, y_train)

# # get features contribution to the model
# RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#                        max_depth=None, max_features='auto', max_leaf_nodes=None,
#                        min_impurity_decrease=0.0,
#                        min_samples_leaf=1, min_samples_split=2,
#                        min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
#                        oob_score=False, random_state=None, verbose=0,
#                        warm_start=False, )
# columns = train_data.drop(['Segmentation'], axis=1).columns
#
# feature_imp = pd.Series(rfc.feature_importances_, index=columns).sort_values(ascending=False)
# print(feature_imp)
#
# # Creating a bar plot
# sns.barplot(x=feature_imp, y=feature_imp.index)
# # Add labels to your graph
# plt.xlabel('Feature Importance Score')
# plt.ylabel('Features')
# plt.title("Visualizing Important Features")
# plt.legend()
# plt.show()

# make predictions on split test set
y_pred = rfc.predict(x_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# create a random forest classifier
rfc = RandomForestClassifier(n_estimators=20, max_features="auto", random_state=23)

# train the model using the split training set
rfc.fit(x, y)

# make predictions on split test set
y_pred = rfc.predict(test_data)

write_to_csv(ids, '../predictions/predictedFromRF.csv', y_pred)
