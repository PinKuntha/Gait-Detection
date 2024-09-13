import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt


# Loading labelling data set
DataLabelPath = "Datasets/Datalabelling.csv"
dataset = pd.read_csv(DataLabelPath)
#axis_data = dataset.iloc[:,1:4]
#print(dataset.iloc[:,1:4]) # only x, y, and z


#_______________________________Extract Statistic Features________________________________

#1. z_values: combine three axiss together(acc_x, acc_y, acc_z)
z_values = (dataset["AccX_Filtered"]**2 + dataset["AccY_Filtered"]**2 + dataset["AccZ_Filtered"]**2)**0.5
dataset["z_values"] = z_values

# 2. Coefficient (acc_x, acc_y, acc_z)
cv = lambda x: np.std(x, ddof=1) / np.mean(x) * 100 
dataset['coef_var'] = dataset.apply(cv, axis=1)

# 3. mean
dataset['mean'] = dataset.apply(lambda row: row[['AccX_Filtered', 'AccY_Filtered', 'AccZ_Filtered']].mean(), axis=1)

# 4. standard deviation
dataset['std'] = dataset.apply(lambda row: row[['AccX_Filtered', 'AccY_Filtered', 'AccZ_Filtered']].std(), axis=1)

# 5. Range
def calculate_range(row):
    return row.max() - row.min()
dataset['range'] = dataset[['AccX_Filtered', 'AccY_Filtered', 'AccZ_Filtered']].apply(calculate_range, axis=1)

# 6.  Median
dataset['median'] = dataset.apply(lambda row: row[['AccX_Filtered', 'AccY_Filtered', 'AccZ_Filtered']].median(), axis=1)
#dic_features_ext = {"z_values": }
feature_extracted = pd.DataFrame(dataset.iloc[:, 5:])
feature_extracted["labels"] = dataset["labels"]
feature_extracted.to_csv("Datasets/ExtractedFeaturesWithLabelling.csv", index=False)
print("feaure_extracted: ", feature_extracted)

plt.plot(dataset["z_values"], '-', color='blue', label='z_values')
#plt.plot(dataset['coef_var'], "-", color='red', label='cv')
plt.plot(dataset['mean'], "-", color='black', label='mean')
plt.plot(dataset['std'], "-", color='yellow', label='std')
plt.plot(dataset['range'], "-", color='green', label='rage')
plt.plot(dataset['median'], "-", color='purple', label='median')
plt.plot(dataset['labels'], "-", color='orange', label='labelling')
plt.legend()
plt.title('Filtered, Detection Peak and Labelling')
plt.show()

#___________________________________________________________________________________________________________________

#print("dataset: ", dataset)