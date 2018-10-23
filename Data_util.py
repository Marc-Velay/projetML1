import pandas as pd
from collections import Counter
import collections
import numpy as np
import tensorflow as tf


def read_data(fname):
    data = pd.read_csv(fname,
                        names=["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Marital Status",
                        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
                        "Hours per week", "Country", "Target"],
                        sep=r'\s*,\s*',
                        engine='python',
                        na_values="?")
    return data

def filter_nan(data):
    df = pd.DataFrame()
    for index, row in data.iterrows():
        if(not any(row.isnull())):
            df = df.append(row)
    return df

def int2onehot(list_len, index):
    one_hot = np.zeros((1, list_len))
    one_hot[np.arange(1), index] = 1

    return list(one_hot[0])

def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


def remap(x, in_min, in_max, out_min, out_max):
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def normalise_data(X, row_names):
    for rname in row_names:
        X[rname] = remap(X[rname], X[rname].min(), X[rname].max(), 0, 1)
    return X

def class2vect(data):
    new_data = []
    targets = []
    for index, row in data.iterrows():
        col_list = []
        if(not any(row.isnull())):
            col_list.append(row["Age"])
            col_list.append(int2onehot(len(WORK_DICT), WORK_DICT.index(row["Workclass"])))
            #col_list.append(WORK_DICT.index(row["Workclass"]))
            col_list.append(row["fnlwgt"])
            col_list.append(int2onehot(len(EDU_DICT), EDU_DICT.index(row["Education"])))
            #col_list.append(EDU_DICT.index(row["Education"]))
            col_list.append(row["Education-Num"])
            col_list.append(int2onehot(len(MARITAL_DICT), MARITAL_DICT.index(row["Marital Status"])))
            col_list.append(int2onehot(len(OCCUP_DICT), OCCUP_DICT.index(row["Occupation"])))
            col_list.append(int2onehot(len(RELATIONSHIP_DICT), RELATIONSHIP_DICT.index(row["Relationship"])))
            col_list.append(int2onehot(len(RACE_DICT), RACE_DICT.index(row["Race"])))
            #col_list.append(MARITAL_DICT.index(row["Marital Status"]))
            #col_list.append(OCCUP_DICT.index(row["Occupation"]))
            #col_list.append(RELATIONSHIP_DICT.index(row["Relationship"]))
            #col_list.append(RACE_DICT.index(row["Race"]))
            col_list.append(row["Capital Gain"])
            col_list.append(row["Capital Loss"])
            col_list.append(row["Hours per week"])
            col_list.append(int2onehot(len(COUNTRY_DICT), COUNTRY_DICT.index(row["Country"])))
            #col_list.append(COUNTRY_DICT.index(row["Country"]))
            col_list = flatten(col_list)
            new_data.append(col_list)

            targets.append(int2onehot(len(TARGET_DICT),TARGET_DICT.index(row["Target"])))
    return new_data, targets

def get_unique_elems(col):
    return set(col)


#print("WORKCLASS", Counter(list(data["Workclass"])))
#print("EDUCATION", Counter(list(data["Education"])))
#print("EDU-NUM", Counter(list(data["Education-Num"])))
#print("MARITAL STATUS", Counter(list(data["Martial Status"])))

WORK_DICT = ['Federal-gov', 'Self-emp-inc', 'State-gov', 'Local-gov', 'Without-pay', 'Private', 'Self-emp-not-inc', 'Never-worked']
EDU_DICT = ['HS-grad', '1st-4th', 'Preschool', 'Assoc-acdm', 'Prof-school', '5th-6th', 'Bachelors', 'Assoc-voc', '10th', '7th-8th', '11th', 'Some-college', 'Masters', '9th', 'Doctorate', '12th']
MARITAL_DICT = ['Separated', 'Divorced', 'Never-married', 'Married-spouse-absent', 'Married-civ-spouse', 'M', 'Widowed', 'Married-AF-spouse']
OCCUP_DICT = ['Other-service', 'Transport-moving', 'Adm-clerical', 'Machine-op-inspct', 'Armed-Forces', 'Exec-managerial', 'Farming-fishing', 'Sales', 'Protective-serv', 'Priv-house-serv', 'Tech-support', None, 'Handlers-cleaners', 'Prof-specialty', 'Craft-repair']
RELATIONSHIP_DICT = ['Unmarried', 'Not-in-family', 'Wife', 'Own-child', 'Other-relative', None, 'Husband']
RACE_DICT = ['Other', None, 'White', 'Amer-Indian-Eskimo', 'Black', 'Asian-Pac-Islander']
SEX_DICT = ['Male', 'Female', None]
COUNTRY_DICT = ['Greece', 'Cambodia', 'Poland', 'Mexico', 'France', 'Cuba', 'Outlying-US(Guam-USVI-etc)', 'Thailand', 'Germany', 'Ecuador', 'Guatemala', 'Philippines', 'Honduras', 'Japan', 'Vietnam', 'Holand-Netherlands', 'Hungary', 'Nicaragua', 'Scotland', 'England', 'South', 'Puerto-Rico', 'United-States', 'Jamaica', 'Iran', 'Laos', 'Peru', 'Canada', 'Italy', 'Portugal', 'Taiwan', 'Haiti', 'Trinadad&Tobago', 'El-Salvador', 'Ireland', 'China', 'Yugoslavia', 'Hong', 'Dominican-Republic', 'Columbia', 'India']
TARGET_DICT = ['<=50K', '>50K']
