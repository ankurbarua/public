from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Specify the file names
file1_name = "/content/parties.csv"

# Specify the columns to read from each file
columns_to_read_parties = ['party_sex', 'party_age', 'party_race', 'at_fault',
                           'party_number', 'party_type', 'cellphone_use_type',
                           'party_number_injured', 'party_number_killed',
                           'movement_preceding_collision', 'vehicle_year', 'party_sobriety']

# Read each file into separate DataFrames
df = pd.read_csv(file1_name, usecols=columns_to_read_parties)

# Remove rows with any missing values in the specified columns
df = df.dropna()

# Ensure 'party_age' is numeric
df['party_age'] = pd.to_numeric(df['party_age'], errors='coerce')

# Modify the 'party_sex' column
df['party_sex'] = df['party_sex'].replace({'male': 0, 'female': 1}).astype(int)

# Map the 'party_race' column according to the specified categories
race_mapping = {'white': 0, 'hispanic': 1, 'black': 2, 'asian': 3, 'other': 4}
df['party_race'] = df['party_race'].replace(race_mapping).astype(int)

# Map the 'movement_preceding_collision' according to the categories
# Mapping: {'proceeding straight': 0, 'other': 1, 'stopped': 2, 'slowing/stopping': 3, 'making left turn': 4, 'changing lanes': 5, 'other unsafe turning': 6, 'crossed into opposing lane': 7, 'making right turn': 8, 'backing': 9, 'passing other vehicle': 10, 'entering traffic': 11, 'traveling wrong way': 12, 'ran off road': 13, 'making u-turn': 14, 'merging': 15, 'parking maneuver': 16, 'parked': 17}
df['movement_preceding_collision'] = pd.factorize(df['movement_preceding_collision'])[0]

# Map the 'party_type' according to the categories
# Mapping: {'driver': 0, 'pedestrian': 1, 'bicyclist': 2, 'parked vehicle': 3, 'other': 4}
df['party_type'] = pd.factorize(df['party_type'])[0]

# Map the 'cellphone_use_type' according to the categories
# Mapping: {'cellphone not in use': 0, 'cellphone in use (hands-free)': 1, 'cellphone in use': 2, 'no cellphone/unknown': 3, 'cellphone in use (handheld)': 4}
df['cellphone_use_type'] = pd.factorize(df['cellphone_use_type'])[0]

# Map the 'party_sobriety' according to the categories
# Mapping: {'not applicable': 0, 'impairment unknown': 1, 'had not been drinking': 2, 'had been drinking, under influence': 3, 'had been drinking, not under influence': 4, 'had been drinking, impairment unknown': 5}
df['party_sobriety'] = pd.factorize(df['party_sobriety'])[0]

# Use only 'party_sex' and 'party_age' as features and 'at_fault' as the target
X = df[['party_sex', 'party_age', 'party_race',
        'party_number', 'party_type', 'cellphone_use_type',
        'party_number_injured', 'party_number_killed',
        'movement_preceding_collision', 'vehicle_year', 'party_sobriety']]
y = df['at_fault']

# Split the dataset into 80% train and 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure all feature columns are numeric
X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_test = X_test.apply(pd.to_numeric, errors='coerce')

# Drop any rows that might have NaN values after conversion
X_train = X_train.dropna()
X_test = X_test.dropna()
y_train = y_train[X_train.index]
y_test = y_test[X_test.index]

clf = LogisticRegression()
clf = clf.fit(X_train, y_train)

score = clf.score(X_test, y_test)
print(f"The accuracy for LogisticRegression is: {score * 100}%")

