import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from utilies import sk_util as sk

#_______________________________________________Loading Dataset_________________________________________
# Load the data from a CSV file using pandas
df = pd.read_csv('Datasets/combined_data.csv')  
# sk_dataFrame.columns = ['ID', 'User ID', 'strDate', 'NameDevice', 'AccX', 'AccY', 'AccZ',
#                         'GyroX', 'GyroY', 'GyroZ', 'AngleX', 'AngleY', 'AngleZ', 'MagX', 'MagY', 'MagZ']
df = pd.DataFrame(df)

#______________________________________________Peak Detection___________________________________________

# create function for detecting low peaks(local maxima) based on high peak of using scipy
def local_min_peaks(acc_data, r_peak_indexs):
    local_mins=[]
    first_window =  acc_data[0:r_peak_indexs[0]]
    local_mins.append(np.argmin(first_window))
    for i in range(0, len(r_peak_indexs)-1):
        
        window_length = r_peak_indexs[i+1] - r_peak_indexs[i]
        #print(window_length)
        #print("i: ", i)
        #print("len: ", len(r_peak_index))
        
        window_data = acc_data[r_peak_indexs[i]:r_peak_indexs[i+1]] #old
        #local_idxs = argrelmin(window_data)  #
        local_idxs = np.argmin(window_data) #local index
        real_idx = local_idxs+ r_peak_indexs[i] # global index
        local_mins.append(real_idx) 
        
    return local_mins

## Smooth and   the necessary featrues by gaussian_filter1d
_col_list = ['AccX', 'AccY', 'AccZ']
data = sk.dataset_signal_filtering(
        dataset=df,
        cols=_col_list)

# _col_list_ = ['GyroX', 'GyroY', 'GyroZ', 'AngleX']
# data1 = sk.dataset_signal_bandpass_filtering(
#         dataset=df,
#         cols_=_col_list_)
# print(data)
data = pd.DataFrame(data)
vertical_accF = data['AccY_Filtered']

#accF_indices = np.abs(vertical_accF-1)<0.01  # I add this part
#vertical_accF[accF_indices] = 0             # this one also

vertical_accF = (vertical_accF-1)*10  # change this one according to patients data(0 when no walking, but healthy 1 when no walking)

sk_up_peaksF, _ = find_peaks(vertical_accF, distance=10, height=2.5, prominence =3) # before operate -1
#sk_up_peaksF, _ = find_peaks(vertical_accF, distance=10, height=3)  # after operate -1
sk_down_peaksF = local_min_peaks(vertical_accF,sk_up_peaksF)

#plot graph and peak of data filtered
'''
plt.plot(vertical_acc, '-', color='black', label='Original')
plt.plot(vertical_accF, '-', color='blue', label='Filtered')
plt.plot(sk_up_peaksF, vertical_accF[sk_up_peaksF],
          "x", color='red', label='Highest Peaks')
#plt.plot(sk_midSwing_peaks, vertical_acc[sk_midSwing_peaks],
#          "o", color='blue', label='Middle Peaks')
plt.plot(sk_down_peaksF, vertical_accF[sk_down_peaksF],
          "o", color='green', label='Lowest Peaks')
plt.legend()
plt.title('Detection Peak and Filtered')
plt.show()
'''

#______________________________________________Define Labeling___________________________________________
# Define the labels for each sample in the data based on the number of peaks
# Define the labels for the two classes (walking and non-walking)
walking_label = 40
non_walking_label = 0
#data.shape[1]

data["labels"] = np.zeros(len(data))
for i in range(len(sk_up_peaksF)-1):
    if sk_up_peaksF[i+1] - sk_up_peaksF[i] <= 100: # distance
        df.iloc[sk_up_peaksF[i]:sk_up_peaksF[i+1], data.shape[1]-1:data.shape[1]] = walking_label
for i in range(len(sk_down_peaksF)-1):
    if sk_down_peaksF[i+1] - sk_down_peaksF[i] <= 100:
        df.iloc[sk_down_peaksF[i]:sk_down_peaksF[i+1], data.shape[1]-1:data.shape[1]] = walking_label

# plt.plot(vertical_acc, '-', color='black', label='Original')
plt.plot(vertical_accF, '-', color='blue', label='Filtered')
#plt.plot(df["labels"], "-", color='red', label='Labeling')
plt.plot(sk_up_peaksF, vertical_accF[sk_up_peaksF],
         "x", color='red', label='Highest Peaks')
plt.plot(sk_down_peaksF, vertical_accF[sk_down_peaksF],
         "o", color='green', label='Lowest Peaks')
plt.legend()
plt.title('High_Peaks and Down_Peaks Detection')
plt.show()
print(data)
df.to_csv("Datasets/Datalabelling.csv", columns=["AccX_Filtered", "AccY_Filtered", "AccZ_Filtered", "labels"])
#df.to_csv("Datasets/Datalabelling.csv", columns=["GyroX_Filtered", "GyroY_Filtered", "GyroZ_Filtered", "labels"], index=False)

























#     peaks, _ = find_peaks(stride["GyroZ"], distance=50, height=30)
#     if len(peaks) > 1:
#         labels.append(1) # walking
#     else:
#         labels.append(0) # non-walking
#     print(f"Peak:{peaks}")
#     strides.append(stride)

# print(labels)
# labels = []
# for stride in strides:
#     print(stride["GyroZ"])
#     # print(stride)
#     # Find the peaks in the vertical acceleration signal
#     peaks, _ = find_peaks(stride["GyroZ"], distance=50, height=30)
#     # print(f"Peak:{peaks}")
#     # Determine the label based on the number of peaks
#     if len(peaks) > 1:
#         labels.append(1) # walking
#     else:
#         labels.append(0) # non-walking

# print(f"Labelling: {labels}")
# labels_of_HS = []

# for i in range(len(vertical_accF)):
#     if i in sk_up_peaksF:
#         labels_of_HS.append(walking_label)
#     elif i in sk_down_peaksF:
#         labels_of_HS.append(walking_label)
#     else:
#         labels_of_HS.append(non_walking_label)

# vertical_accF_values_reshaped = vertical_accF.values.reshape(-1, 1)
# labels_of_HS_reshaped = np.array(foot_off_labels)
# labeled_signal = np.concatenate([vertical_accF_values_reshaped, labels_of_HS_reshaped.reshape(-1, 1)], axis=1)

# df_ = pd.DataFrame(labeled_signal)
# df_.columns = ["data","labeling"]
# df_.to_csv("Datasets/RultTrain1.csv")
# print(labeled_signal.shape)