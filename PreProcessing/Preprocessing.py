from statistics import median
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing

from helper_functions import replace_nulls_with_mode, replace_nulls_with_median, replace_nulls, \
    print_correlation_matrix, \
    standardize_data

pd.set_option('display.max_columns', None)


def preprocessing(mode):
    # Date Reading
    train_data = pd.read_csv('../data/train.csv')
    test_data = pd.read_csv('../data/test.csv')

    # 1-Filter input features
    # print(data['Holiday'].unique()) #['No Holiday' 'Holiday']
    train_data['Gender'] = train_data['Gender'].replace(['Male'], 1)
    train_data['Gender'] = train_data['Gender'].replace(['Female'], 0)
    test_data['Gender'] = test_data['Gender'].replace(['Male'], 1)
    test_data['Gender'] = test_data['Gender'].replace(['Female'], 0)

    train_data['Ever_Married'] = train_data['Ever_Married'].replace(['Yes'], 1)
    train_data['Ever_Married'] = train_data['Ever_Married'].replace(['No'], 0)
    test_data['Ever_Married'] = test_data['Ever_Married'].replace(['Yes'], 1)
    test_data['Ever_Married'] = test_data['Ever_Married'].replace(['No'], 0)

    train_data['Graduated'] = train_data['Graduated'].replace(['Yes'], 1)
    train_data['Graduated'] = train_data['Graduated'].replace(['No'], 0)
    test_data['Graduated'] = test_data['Graduated'].replace(['Yes'], 1)
    test_data['Graduated'] = test_data['Graduated'].replace(['No'], 0)

    train_data['Spending_Score'] = train_data['Spending_Score'].replace(['Low'], 0)
    train_data['Spending_Score'] = train_data['Spending_Score'].replace(['Average'], 1)
    train_data['Spending_Score'] = train_data['Spending_Score'].replace(['High'], 2)
    test_data['Spending_Score'] = test_data['Spending_Score'].replace(['Low'], 0)
    test_data['Spending_Score'] = test_data['Spending_Score'].replace(['Average'], 1)
    test_data['Spending_Score'] = test_data['Spending_Score'].replace(['High'], 2)

    if (mode == 2):
        print(train_data.isnull().sum(axis=0))
        print(test_data.isnull().sum(axis=0))
        train_data, test_data = replace_nulls(train_data, test_data)
    ''' 
    OneHotEncoder_gender = pd.get_dummies(train_data['Gender'], prefix='Gender')
    train_data = train_data.join(OneHotEncoder_gender)
    train_data.drop('Gender', axis=1, inplace=True)

    OneHotEncoder_gender = pd.get_dummies(test_data['Gender'], prefix='Gender')
    test_data = test_data.join(OneHotEncoder_gender)
    test_data.drop('Gender', axis=1, inplace=True)

    OneHotEncoder_grad = pd.get_dummies(train_data['Graduated'], prefix='Graduated')
    train_data = train_data.join(OneHotEncoder_grad)
    train_data.drop('Graduated', axis=1, inplace=True)

    OneHotEncoder_grad = pd.get_dummies(test_data['Graduated'], prefix='Graduated')
    test_data = test_data.join(OneHotEncoder_grad)
    test_data.drop('Graduated', axis=1, inplace=True)'''

    OneHotEncoder_prof = pd.get_dummies(train_data['Profession'], prefix='Profession')
    train_data = train_data.join(OneHotEncoder_prof)
    train_data.drop(['Profession'], axis=1, inplace=True)

    OneHotEncoder_prof = pd.get_dummies(test_data['Profession'], prefix='Profession')
    test_data = test_data.join(OneHotEncoder_prof)
    test_data.drop(['Profession'], axis=1, inplace=True)

    OneHotEncoder_var = pd.get_dummies(train_data['Var_1'], prefix='Var_1')
    train_data = train_data.join(OneHotEncoder_var)
    train_data.drop(['Var_1'], axis=1, inplace=True)

    OneHotEncoder_var = pd.get_dummies(test_data['Var_1'], prefix='Var_1')
    test_data = test_data.join(OneHotEncoder_var)
    test_data.drop(['Var_1'], axis=1, inplace=True)

    # REPLACE NULLS WITH THE MODE VALUE OF THE COLUMN.
    if (mode == 1):
        train_data = replace_nulls_with_mode(train_data)
        test_data = replace_nulls_with_mode(test_data)
    else:
        train_data = replace_nulls_with_median(train_data)
        test_data = replace_nulls_with_median(test_data)
    print(train_data.isnull().sum(axis=0))
    print(test_data.isnull().sum(axis=0))
    # PRINT CORRELATION MATRIX
    # print_correlation_matrix(train_data)

    return train_data, test_data


def applyPCA(train_data, labels):
    standardized_data = standardize_data(train_data)
    pca = PCA(n_components=20)
    principalComponents = pca.fit_transform(standardized_data)
    principalDf = pd.DataFrame(data=principalComponents, )
    print(principalDf)
    # finalDf = pd.concat([principalDf, labels], axis=1)
    return principalDf
    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(1, 1, 1)
    # ax.set_xlabel('Principal Component 1', fontsize=15)
    # ax.set_ylabel('Principal Component 2', fontsize=15)
    # ax.set_title('2 component PCA', fontsize=20)
    # targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
    # colors = ['r', 'g', 'b']
    # for target, color in zip(targets, colors):
    #     indicesToKeep = finalDf['Segmentation'] == target
    #     ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
    #                , finalDf.loc[indicesToKeep, 'principal component 2']
    #                , c=color
    #                , s=50)
    # ax.legend(targets)
    # ax.grid()
    # pca.explained_variance_ratio_
