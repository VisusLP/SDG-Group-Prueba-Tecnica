import pandas as pd
"""
This script loads a dataset from a CSV file, takes a random sample of 10% of the data,
and saves the sample to a new CSV file.
"""

data_path = '/root/dags/data/dataset.csv'
df = pd.read_csv(data_path, sep=';', decimal=',')
print("Dataset loaded.")

sample_df = df.sample(frac=0.1)
sample_path = '/root/dags/data/dataset_sample.csv'
sample_df.to_csv(sample_path, sep=';', decimal=',', index=False)
print("Sample dataset saved.")