import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sn
from sklearn.preprocessing import OneHotEncoder


def preprocessing():
    # Date Reading
    train_data = pd.read_csv('../data/train.csv')
    test_data = pd.read_csv('../data/test.csv')

    # 1-Filter input features
    # print(data['Holiday'].unique()) #['No Holiday' 'Holiday']
    train_data['Gender'] = train_data['Gender'].replace(['Male'], 1)
    train_data['Gender'] = train_data['Gender'].replace(['Female'], 0)

    train_data['Ever_Married'] = train_data['Ever_Married'].replace(['Yes'], 1)
    train_data['Ever_Married'] = train_data['Ever_Married'].replace(['No'], 0)

    train_data['Graduated'] = train_data['Graduated'].replace(['Yes'], 1)
    train_data['Graduated'] = train_data['Graduated'].replace(['No'], 0)

    one_hot_encoder = OneHotEncoder(sparse=False, handle_unknown='error', drop='first')
    OneHotEncoder_prof = pd.DataFrame(one_hot_encoder.fit_transform(train_data[['Profession']]))
    train_data.drop(['Profession'], axis=1,inplace=True)
    train_data=train_data.join(OneHotEncoder_prof)

    print(train_data.head())

    print(train_data.isnull().sum(axis=0))
    return train_data, test_data
