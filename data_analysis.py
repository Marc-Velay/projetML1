import pandas as pd
from collections import Counter

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



def class2int(data):
    new_data = []
    for index, row in data.iterrows():
        col_list = []
        if(not any(row.isnull())):
            col_list.append(row["Age"])
            col_list.append(WORK_DICT.index(row["Workclass"]))
            col_list.append(row["fnlwgt"])
            col_list.append(EDU_DICT.index(row["Education"]))
            col_list.append(row["Education-Num"])
            col_list.append(MARITAL_DICT.index(row["Marital Status"]))
            col_list.append(OCCUP_DICT.index(row["Occupation"]))
            col_list.append(RELATIONSHIP_DICT.index(row["Relationship"]))
            col_list.append(RACE_DICT.index(row["Race"]))
            col_list.append(row["Capital Gain"])
            col_list.append(row["Capital Loss"])
            col_list.append(row["Hours per week"])
            col_list.append(COUNTRY_DICT.index(row["Country"]))
            print(col_list)
            target = TARGET_DICT.index(row["Target"])
            print(target)
            input()
    return new_data

#pprint(data)

#print(data["Workclass"])

#print(data.isnull())

#print("WORKCLASS", Counter(list(data["Workclass"])))
#print("EDUCATION", Counter(list(data["Education"])))
#print("EDU-NUM", Counter(list(data["Education-Num"])))
#print("MARITAL STATUS", Counter(list(data["Martial Status"])))

WORK_DICT = ['Self-emp-not-inc', 'Local-gov', 'Federal-gov', 'Without-pay', 'State-gov', 'Private', 'Self-emp-inc']
EDU_DICT = ['7th-8th', 'Assoc-acdm', '9th', 'Some-college', 'Bachelors', '5th-6th', '1st-4th', 'HS-grad', 'Assoc-voc', 'Doctorate', 'Preschool', '12th', 'Masters', 'Prof-school', '10th', '11th']
MARITAL_DICT = ['Separated', 'Divorced', 'Never-married', 'Married-spouse-absent', 'Married-civ-spouse', 'M', 'Widowed', 'Married-AF-spouse']
OCCUP_DICT = ['Other-service', 'Transport-moving', 'Adm-clerical', 'Machine-op-inspct', 'Armed-Forces', 'Exec-managerial', 'Farming-fishing', 'Sales', 'Protective-serv', 'Priv-house-serv', 'Tech-support', None, 'Handlers-cleaners', 'Prof-specialty', 'Craft-repair']
RELATIONSHIP_DICT = ['Unmarried', 'Not-in-family', 'Wife', 'Own-child', 'Other-relative', None, 'Husband']
RACE_DICT = ['Other', None, 'White', 'Amer-Indian-Eskimo', 'Black', 'Asian-Pac-Islander']
SEX_DICT = ['Male', 'Female', None]
COUNTRY_DICT = ['Germany', 'Columbia', 'Taiwan', 'Thailand', 'United-States', 'India', 'Greece', 'Ireland', 'Iran', 'Portugal', 'Laos', 'Philippines', 'Puerto-Rico', 'Guatemala', 'Yugoslavia', 'England', 'Poland', 'Canada', 'Jamaica', 'Italy', 'Japan', 'China', 'Outlying-US(Guam-USVI-etc)', 'France', 'Mexico', 'Dominican-Republic', 'South', 'Haiti', 'Trinadad&Tobago', 'El-Salvador', 'Honduras', 'Vietnam', 'Nicaragua', None, 'Ecuador', 'Peru', 'Cambodia', 'Scotland', 'Cuba', 'Hong']
TARGET_DICT = ['>50K', '<=50K']


def get_unique_elems(col):
    return set(col)



if __name__ == "__main__":
    data = read_data("data/adult.data")
    #filter_nan(data)

    #for col in data:
    #    print(get_unique_elems(data[col]))
    new_data = class2int(data)
