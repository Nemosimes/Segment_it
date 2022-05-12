from statistics import median

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sn
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing

pd.set_option('display.max_columns', None)


def preprocessing():
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

    # print(train_data.isnull().sum(axis=0))
    # REPLACE NULLS WITH THE MEDIAN VALUE OF THE COLUMN.
    train_data['Ever_Married'] = train_data['Ever_Married'].fillna(median(train_data['Ever_Married']))
    train_data['Graduated'] = train_data['Graduated'].fillna(median(train_data['Graduated']))
    train_data['Work_Experience'] = train_data['Work_Experience'].fillna(median(train_data['Work_Experience']))
    train_data['Family_Size'] = train_data['Family_Size'].fillna(median(train_data['Family_Size']))
    test_data['Ever_Married'] = test_data['Ever_Married'].fillna(median(test_data['Ever_Married']))
    test_data['Graduated'] = test_data['Graduated'].fillna(median(test_data['Graduated']))
    test_data['Work_Experience'] = test_data['Work_Experience'].fillna(median(test_data['Work_Experience']))
    test_data['Family_Size'] = test_data['Family_Size'].fillna(median(test_data['Family_Size']))

    return train_data, test_data


preprocessing()
