# initializing all the base model objects with default parameters
from statistics import mean

from matplotlib import pyplot
from numpy import std
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestRegressor, StackingClassifier, RandomForestClassifier, \
    GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from PreProcessing.Preprocessing import preprocessing
from helper_functions import write_to_csv, get_model_data


def get_models():
    models = dict()
    models['lr'] = LogisticRegression()
    models['knn'] = KNeighborsClassifier()
    models['cart'] = DecisionTreeClassifier()
    models['svm'] = SVC()
    models['bayes'] = GaussianNB()
    models['rfc'] = RandomForestClassifier(n_estimators=100, max_features="auto", random_state=23)
    models['dtc'] = DecisionTreeClassifier(criterion='entropy', random_state=23)
    models['gbc'] = GradientBoostingClassifier(n_estimators=100, max_features="auto", random_state=23)
    models['stacking'] = get_stacking()
    return models


def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=23)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores


# get a stacking ensemble of models
def get_stacking():
    # define the base models
    level0 = list()
    level0.append(('lr', LogisticRegression()))
    level0.append(('knn', KNeighborsClassifier()))
    level0.append(('cart', DecisionTreeClassifier()))
    level0.append(('svm', SVC()))
    level0.append(('bayes', GaussianNB()))
    level0.append(('rfc', RandomForestClassifier(n_estimators=100, max_features="auto", random_state=23)))
    level0.append(('dtc', DecisionTreeClassifier(criterion='entropy', random_state=23)))
    level0.append(('gbc', GradientBoostingClassifier(n_estimators=100, max_features="auto", random_state=23)))
    # define meta learner model
    level1 = LogisticRegression()
    # define the stacking ensemble
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=5)
    return model


train_data, test_data = preprocessing(1)
x, y, ids, test_data = get_model_data(train_data, test_data)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)
model = get_stacking()
model.fit(x_train, y_train)

# make predictions on split test set
y_pred = model.predict(x_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# create a random forest classifier
model = get_stacking()
model.fit(x, y)

# make predictions on split test set
y_pred = model.predict(test_data)

write_to_csv(ids, '../predictions/predictedFromStacking.csv', y_pred)

# models = get_models()
# # evaluate the models and store results
# results, names = list(), list()
# for name, model in models.items():
#     scores = evaluate_model(model, x, y)
#     results.append(scores)
#     names.append(name)
#     print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# # plot model performance for comparison
# pyplot.boxplot(results, labels=names, showmeans=True)
# pyplot.show()

# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)
#
# model_1 = SVC(kernel='linear')
# model_2 = DecisionTreeClassifier(criterion='entropy', random_state=23)
# model_3 = RandomForestRegressor()
#
# # putting all base model objects in one list
# all_models = [model_1, model_2, model_3]
#
# # computing the stack features
# s_train, s_test = StackingClassifier(all_models, x_train, x_test,
#                                      y_train, regression=True, n_folds=4)
#
# # initializing the second-level model
# final_model = model_1
#
# # fitting the second level model with stack features
# final_model = final_model.fit(s_train, y_train)
#
# # predicting the final output using stacking
# pred_final = final_model.predict(x_test)
#
# # printing the root mean squared error between real value and predicted value
# print(mean_squared_error(y_test, pred_final))  # initializing all the base model objects with default parameters
