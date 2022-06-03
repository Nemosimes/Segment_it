
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from PreProcessing.Preprocessing import preprocessing
from helper_functions import write_to_csv,get_model_data,decision_boundary

train_data, test_data = preprocessing(1)
train_data['Segmentation'] = train_data['Segmentation'].replace(['A'], 3)
train_data['Segmentation'] = train_data['Segmentation'].replace(['B'], 2)
train_data['Segmentation'] = train_data['Segmentation'].replace(['C'], 1)
train_data['Segmentation'] = train_data['Segmentation'].replace(['D'], 0)
x, y, ids, test_data = get_model_data(train_data, test_data)

x_train, x_test, y_train, y_test,  = train_test_split(x, y, test_size=0.2, random_state=23)

# create a random forest classifier
rfc = RandomForestClassifier(n_estimators=100, max_features="sqrt",criterion='gini',max_depth= None, min_samples_split=50,min_samples_leaf=5, random_state=23)# train the model using the split training set
rfc.fit(x_train, y_train)

# make predictions on split test set
y_pred = rfc.predict(x_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
#decision_boundary(x,y,classifier)

# create a random forest classifier
rfc = RandomForestClassifier(n_estimators=100,criterion='gini', max_features="auto", random_state=23)

# train the model using the split training set
rfc.fit(x, y)

# make predictions on split test set
y_pred = rfc.predict(test_data)

write_to_csv(ids, '../predictions/predictedFromRF.csv', y_pred)
