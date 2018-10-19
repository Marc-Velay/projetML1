import pandas as pd
from collections import Counter

def read_data(fname):
    data = pd.read_csv(fname,
                        names=["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
                        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
                        "Hours per week", "Country", "Target"],
                        sep=r'\s*,\s*',
                        engine='python',
                        na_values="?")

    return data

def filter_nan(data):
    non_null = 0
    df = pd.DataFrame()
    for index, row in data.iterrows():
        if(not any(row.isnull())):
            non_null +=1
            df = df.append(row)

    print(data.shape[0]-non_null, " nulls out of ", data.shape[0])
    print(df.shape)


#pprint(data)

#print(data["Workclass"])

#print(data.isnull())

#print("WORKCLASS", Counter(list(data["Workclass"])))
#print("EDUCATION", Counter(list(data["Education"])))
#print("EDU-NUM", Counter(list(data["Education-Num"])))
#print("MARITAL STATUS", Counter(list(data["Martial Status"])))



if __name__ == "__main__":
    data = read_data("data/adult.data")
    filter_nan(data)
