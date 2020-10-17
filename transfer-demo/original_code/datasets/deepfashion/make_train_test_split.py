import pandas as pd
from sklearn.model_selection import train_test_split
ddf = pd.read_csv("data.csv")
train_groups, test_groups = train_test_split(ddf, test_size=0.1, random_state=42)
train_groups.to_csv("data_train.csv", index=False)
test_groups.to_csv("data_test.csv", index=False)
