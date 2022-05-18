# get a voting ensemble of models
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from PreProcessing.Preprocessing import preprocessing
from helper_functions import write_to_csv, get_model_data


def get_voting():
    # define the base models
    models = list()
    models.append(('knn1', KNeighborsClassifier(n_neighbors=1)))
    models.append(('knn3', KNeighborsClassifier(n_neighbors=3)))
    models.append(('knn5', KNeighborsClassifier(n_neighbors=5)))
    models.append(('knn7', KNeighborsClassifier(n_neighbors=7)))
    models.append(('knn9', KNeighborsClassifier(n_neighbors=9)))
    models.append(('rfc', RandomForestClassifier(n_estimators=100, max_features="auto", random_state=23)))
    models.append(('dtc', DecisionTreeClassifier(criterion='entropy', random_state=23)))
    models.append(('gbc', GradientBoostingClassifier(n_estimators=100, max_features="auto", random_state=23)))
    # define the voting ensemble
    ensemble = VotingClassifier(estimators=models, voting='hard')
    return ensemble


# get a list of models to evaluate
def get_models():
    models = dict()
    models['knn1'] = KNeighborsClassifier(n_neighbors=1)
    models['knn3'] = KNeighborsClassifier(n_neighbors=3)
    models['knn5'] = KNeighborsClassifier(n_neighbors=5)
    models['knn7'] = KNeighborsClassifier(n_neighbors=7)
    models['knn9'] = KNeighborsClassifier(n_neighbors=9)
    models['rfc'] = RandomForestClassifier(n_estimators=100, max_features="auto", random_state=23)
    models['dtc'] = DecisionTreeClassifier(criterion='entropy', random_state=23)
    models['gbc'] = GradientBoostingClassifier(n_estimators=100, max_features="auto", random_state=23)
    models['hard_voting'] = get_voting()
    return models


train_data, test_data = preprocessing(1)
x, y, ids, test_data = get_model_data(train_data, test_data)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)
model = get_voting()
model.fit(x_train, y_train)

# make predictions on split test set
y_pred = model.predict(x_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# create a random forest classifier
model = get_voting()
model.fit(x, y)

# make predictions on split test set
y_pred = model.predict(test_data)

write_to_csv(ids, '../predictions/predictedFromVoting.csv', y_pred)
