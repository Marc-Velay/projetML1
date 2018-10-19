import pandas as pd
from collections import Counter

data = pd.read_csv("data/adult.data",
                    names=["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
                    "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
                    "Hours per week", "Country", "Target"],
                    sep=r'\s*,\s*',
                    engine='python',
                    na_values="?")
#pprint(data)

#print(data["Workclass"])

#print(data.isnull())

print(Counter(list(data["Workclass"])))
print(Counter(list(data["Workclass"])))
print(Counter(list(data["Workclass"])))

'''
for index, row in data.iterrows():
    print(row)
    input()
'''
