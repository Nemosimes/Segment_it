import math
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error
from PreProcessing.Preprocessing import preprocessing
from helper_functions import write_to_csv,decision_boundary

# importing datasets
train_data, test_data = preprocessing(1)

# data
IDs = test_data["ID"]
train_data = train_data.drop(['ID'], axis=1)
test_data = test_data.drop(['ID'], axis=1)
train_data['Segmentation'] = train_data['Segmentation'].replace(['A'], 3)
train_data['Segmentation'] = train_data['Segmentation'].replace(['B'], 2)
train_data['Segmentation'] = train_data['Segmentation'].replace(['C'], 1)
train_data['Segmentation'] = train_data['Segmentation'].replace(['D'], 0)
y = train_data['Segmentation'].values.reshape(-1, 1)
x = train_data.drop(['Segmentation'], axis=1).values

# Fitting Decision Tree classifier to the training set

classifier = DecisionTreeClassifier(criterion='gini',splitter='random', random_state=0)
classifier.fit(x, y)
y_pred = classifier.predict(test_data)
write_to_csv(IDs, '../predictions/predictedFromDecisionTree.csv', y_pred)
#decision_boundary(x,y,classifier)

# Splitting the dataset into training and test set.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Predicting the test set result
classifier = DecisionTreeClassifier(criterion='gini', splitter='random', random_state=0)
# classifier= DecisionTreeClassifier(criterion='entropy', splitter='best', random_state=0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

# Creating the Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Visulaizing the trianing set result
'''from matplotlib.colors import ListedColormap
x_set, y_set = x_train, y_train
x1, x2 = nm.meshgrid(nm.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step  =0.01),
nm.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
mtp.contourf(x1, x2, classifier.predict(nm.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape),
alpha = 0.75, cmap = ListedColormap(('purple','green' )))
mtp.xlim(x1.min(), x1.max())
mtp.ylim(x2.min(), x2.max())
for i, j in enumerate(nm.unique(y_set)):
  mtp.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
        c = ListedColormap(('purple', 'green'))(i), label = j)
mtp.title('Decision Tree Algorithm (Training set)')
mtp.xlabel('Age')
mtp.ylabel('Estimated Salary')
mtp.legend()
mtp.show()'''
print("Acc: ", accuracy_score(y_test, y_pred))
