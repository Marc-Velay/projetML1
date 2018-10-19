import pandas as pd
data = pd.read_csv(“adult.data.txt",
                    names=["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial
                    Status", “Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital
                    Loss", “Hours per week", "Country", "Target"],
                    sep=r'\s*,\s*',
                    engine='python',
                    na_values="?")
data.tail()
