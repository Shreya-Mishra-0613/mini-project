import pandas as pd

#reading data from selected url
df = pd.read_csv("hf://datasets/MTHR/OCEAN/OCEAN-synthetic.csv")

#display first 5 rows of data set
print(df.head())

#save the data set to a local csv file
df.to_csv("local_ocean_synthetic.csv", index=False)

print("Dataset saved to local_ocean_synthetic.csv")
