import pickle
import pandas as pd
from utilies import sk_util as sk
import matplotlib.pyplot as plt
from utilies import sk_util as sk
import numpy as np

sk_pathFiles1 = "Datasets/"

# Test with real-time data
with open('weights/statistic_features/RandomForest_model.pkl', 'rb') as f:
    rf = pickle.load(f)

df_ = pd.read_csv('Datasets/Normal Working/Gait_Analysis_20227613_20230407153849287.csv')
sk_dataFrame = pd.DataFrame(df_)
# sk_dataRight.columns = ['ID', 'UserID', 'strDate', 'NameDevices', 'AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ', 'AngleX', 'AngleY', 'AngleZ', 'MagX', 'MagY', 'MagZ']
# sk_dataRight.columns = ['ID', 'UserID', 'strDate', 'NameDevices', 'AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ', 'AngleX', 'AngleY', 'AngleZ', 'MagX', 'MagY', 'MagZ']

sk_dataLeft, sk_dataRight, sk_sensor = sk.sk_separate_data_2_lr(sk_dataFrame)

sk_sensorName_L = sk_sensor[0]
sk_sensorName_L = sk_sensorName_L.replace(":","")
sk_sensorName_R = sk_sensor[1]
sk_sensorName_R = sk_sensorName_R.replace(":","")

sk_dataFileLeft = f"{sk_pathFiles1}{sk_sensorName_L}_Foot_real.csv"
sk_dataFileRight = f"{sk_pathFiles1}{sk_sensorName_R}_Foot_real.csv"

sk_dataLeft.to_csv(sk_dataFileLeft, columns=['UserID', 'strDate', 'NameDevices', 'AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ', 'AngleX', 'AngleY', 'AngleZ', 'MagX', 'MagY', 'MagZ'], index=False)
sk_dataRight.to_csv(sk_dataFileRight, columns= ['UserID', 'strDate', 'NameDevices', 'AccX', 'AccY', 'AccZ', 'GyroX', 'GyroY', 'GyroZ', 'AngleX', 'AngleY', 'AngleZ', 'MagX', 'MagY', 'MagZ'], index=False)
print(sk_dataLeft, sk_dataRight)

df_ = pd.read_csv(sk_dataFileRight)
new_data = pd.DataFrame(df_)
## Smooth and Filter the necessary featrues by gaussian_filter1d
_col_list = ['AccX', 'AccY', 'AccZ']
data = sk.dataset_signal_filtering(
        dataset=new_data,
        cols=_col_list)



'''
#______________________for row data___________________________
data.to_csv("Datasets/datafiltered.csv", columns=["AccX_Filtered", "AccY_Filtered", "AccZ_Filtered"], index=False)
df_n = pd.read_csv('Datasets/datafiltered.csv')

real_data = rf.predict(new_data)
indices = np.where(real_data==1)[0]
real_data[indices] = 40
plt.figure(figsize=(10, 8))
plt.plot(data["AccY"], '-', color='blue', label='Real Data Y')
plt.plot(data["AccX"], '-', color='green', label='Real Data X')
plt.plot(data["AccZ"], '-', color='black', label='Real Data Z')
plt.plot(real_data, "-", color='red', label='Data Predict')
plt.legend()
plt.title('Predict With Real Data')
plt.show()
'''
#_______________________________Extract Statistic Features________________________________

selected_f = {"AccX_Filtered":data["AccX_Filtered"], "AccY_Filtered":data["AccY_Filtered"], "AccZ_Filtered":data["AccZ_Filtered"]}
new_data = pd.DataFrame(selected_f)

#1. z_values: combine three axiss together(acc_x, acc_y, acc_z)
z_values = (new_data["AccX_Filtered"]**2 + new_data["AccY_Filtered"]**2 + new_data["AccZ_Filtered"]**2)**0.5
new_data["z_values"] = z_values

# 2. Coefficient (acc_x, acc_y, acc_z)


def cv(x):
    return np.std(x) / np.mean(x)
    
new_data['coef_var'] = new_data.apply(lambda row: cv(row[['AccX_Filtered', 'AccY_Filtered', 'AccZ_Filtered']]), axis=1)

cv = lambda x: np.std(x, ddof=1) / np.mean(x) * 100 
new_data['coef_var'] = new_data.apply(cv, axis=1)

# 3. mean
new_data['mean'] = new_data.apply(lambda row: row[['AccX_Filtered', 'AccY_Filtered', 'AccZ_Filtered']].mean(), axis=1)

# 4. standard deviation
new_data['std'] = new_data.apply(lambda row: row[['AccX_Filtered', 'AccY_Filtered', 'AccZ_Filtered']].std(), axis=1)

# 5. Range
def calculate_range(row):
    return row.max() - row.min()
new_data['range'] = new_data[['AccX_Filtered', 'AccY_Filtered', 'AccZ_Filtered']].apply(calculate_range, axis=1)

# 6.  Median
new_data['median'] = new_data.apply(lambda row: row[['AccX_Filtered', 'AccY_Filtered', 'AccZ_Filtered']].median(), axis=1)

# prediction
feature_extracted = pd.DataFrame(new_data.iloc[:, 3:]) # reduce accx,y,x filtered

print("feature_extracted: ", feature_extracted)
real_data = rf.predict(feature_extracted)

'''indices = np.where(real_data==1)[0]
real_data[indices] = 40'''

plt.figure(figsize=(10, 8))
plt.plot(data["AccY_Filtered"], '-', color='blue', label='Real Data Y')
plt.plot(data["AccX_Filtered"], '-', color='green', label='Real Data X')
plt.plot(data["AccZ_Filtered"], '-', color='black', label='Real Data Z')
plt.plot(real_data, "-", color='red', label='Data Predict')
plt.legend()
plt.title('Predict With Real Data')
plt.show()
#___________________________________________________________________________________



# sk.sk_print_score_predict_real(rf, real_data, y_test)