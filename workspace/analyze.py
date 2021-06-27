import pandas as pd

df = pd.read_csv("./result50.csv", header=0)

new_df = df[df["supply_portion"] == 1.0]
print(new_df.info)
