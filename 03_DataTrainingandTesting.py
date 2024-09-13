
#import libraries
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from utilies import sk_util as sk
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
sns.set_style('whitegrid')


# Load and preprocess the signal data
signal_data = pd.read_csv('Datasets/ExtractedFeaturesWithLabelling.csv')

# data and labeling
#X = signal_data.drop('labels', axis=1) # pass all data in all columns except column "labels"

'''
# for raw data
x_data = {"AccX_Filtered": signal_data["AccX_Filtered"], "AccY_Filtered": signal_data["AccY_Filtered"]-1,
          "AccZ_Filtered":signal_data["AccZ_Filtered"] }  #-1 according to label(0: no walking for patient data, 1: no walk for healthy)
X = pd.DataFrame(x_data)
print("signal_data:", X)
'''

# Train models with extracted statistic features
X = signal_data.drop('labels', axis=1)
print("X: ", X)

#print("signal_data: ", signal_data)  # still keep all data
y = signal_data.labels

indices = np.where(y==40)[0]
y[indices] = 1

print(f"'X' shape: {X.shape}")
print(f"'y' shape: {y.shape}")
pipeline = Pipeline([
    ('min_max_scaler', MinMaxScaler()),
    ('std_scaler', StandardScaler())
])

# Train Test Split
start_time1 = datetime.now()
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.2, random_state=42)

#__________________________Train Machine Learning Models ____________________________


#___________________________1. RBF Kernel SVM____________________________
from sklearn.svm import SVC
model1 = SVC(kernel="rbf", gamma=0.7, C=1.0)
model1.fit(X_train1, y_train1)  # train model
print("1. Linear Kernel SVM")
sk.sk_print_score(model1, "RBFKernelSVM", X_train1, y_train1, X_test1, y_test1, train=True) # This function is for training model
sk.sk_print_score(model1, "RBFKernelSVM", X_train1, y_train1, X_test1, y_test1, train=False) # test model if train= False
end_time1 = datetime.now()
print('Duration: {}\n'.format(end_time1 - start_time1))

#___________________________2. Polynomial Kernel SVM__________________________
# This code trains an SVM classifier using a 2nd-degree polynomial kernel.
# The hyperparameter coef0 controls how much the model is influenced by high degree ploynomials
from sklearn.svm import SVC
start_time2 = datetime.now()

# Define SVM model
model2 = SVC(kernel='poly', degree=2, coef0=1)

# 
model2.fit(X_train1, y_train1)
print("2. Polynomial Kernel SVM")
sk.sk_print_score(model2, "PolynomialKernelSVM", X_train1, y_train1, X_test1, y_test1, train=True)
sk.sk_print_score(model2, "PolynomialKernelSVM", X_train1, y_train1, X_test1, y_test1, train=False)
end_time2 = datetime.now()
print('Duration: {}\n'.format(end_time2 - start_time2))

#____________________________3. Random Forest classifier______________________
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
start_time4 = datetime.now()

# Define RF model
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf.fit(X_train1, y_train1)

# Save trained weight and evaluate model performance
print("Random Forest (RF) Model")
sk.sk_print_score(rf,"RandomForest", X_train1, y_train1, X_test1, y_test1, train=True)
sk.sk_print_score(rf,"RandomForest", X_train1, y_train1, X_test1, y_test1, train=False)
end_time4 = datetime.now()
print('Duration: {}\n'.format(end_time4 - start_time4))


#_____________________________4. K-Nearest Neighbors Algorithm________________
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
start_time5 = datetime.now()

# Scale the features using StandardScaler:
scaler = StandardScaler()
X_train_knn = scaler.fit_transform(X_train1)
X_test_knn = scaler.transform(X_test1)

# Define model
knn = KNeighborsClassifier(n_neighbors=3)

# Train model
knn.fit(X_train_knn, y_train1)

# Save trained weight and evaluate model performance
print("K-Nearest Neighbors Algorithm (KNN model)")
sk.sk_print_score(knn, "KNN", X_train_knn, y_train1, X_test_knn, y_test1, train=True)
sk.sk_print_score(knn,"KNN", X_train_knn, y_train1, X_test_knn, y_test1, train=False)
end_time5 = datetime.now()
print('Duration: {}\n'.format(end_time5 - start_time5))

#__________________________________________5. Naive Bayes______________________________________
from sklearn.naive_bayes import GaussianNB

# Define model
naive_bayes = GaussianNB()

# Train model
naive_bayes.fit(X_train1, y_train1)

# Evaluate model
print('Training NaiveBayes model.')
sk.sk_print_score(naive_bayes, "NaiveBayes", X_train1, y_train1, X_test1, y_test1, train=True)
sk.sk_print_score(naive_bayes, "NaiveBayes", X_train1, y_train1, X_test1, y_test1, train=False)

#__________________________________________6. XGBoost______________________________________
import xgboost as xgb

#Define model
xgb_cl = xgb.XGBClassifier()

# Train model
xgb_cl.fit(X_train1, y_train1)

# Evaluate model
print('Training XGBoost model.')
sk.sk_print_score(xgb_cl, "XGBoost", X_train1, y_train1, X_test1, y_test1, train=True)
sk.sk_print_score(xgb_cl, "XGBoost", X_train1, y_train1, X_test1, y_test1, train=False)

#__________________________________________7. Decision Tree______________________________________
from sklearn import tree

#Define model
dt = tree.DecisionTreeClassifier()

# Train model
dt.fit(X_train1, y_train1)

# Evaluate model
print('Training Decision model.')
sk.sk_print_score(dt, "DecisionTree", X_train1, y_train1, X_test1, y_test1, train=True)
sk.sk_print_score(dt, "DecisionTree", X_train1, y_train1, X_test1, y_test1, train=False)

#---------------------------------ROC Curve of All Models-----------------------------------
#loading models
# 1. LKSVM
lksvm_pred = model1.predict(X_test1)

# 2. PKSVM
pksvm_pred = model2.predict(X_test1)

# 3. RF
rf_pred = rf.predict(X_test1)

# 4. KNN
knn_pred = knn.predict(X_test1)

# 5. NB
nb_pred = naive_bayes.predict(X_test1)

# 6. XG-Boost
xgb_pred = xgb_cl.predict(X_test1)

#___________________________Set up plotting area for all models______________________

from sklearn.metrics import roc_curve, roc_auc_score

# Models
classifiers = ["RBF Kernel SVM", "Polynomial Kernel SVM", "Random Forest", "KNN", "Naive Bayes", "XGBoost", "Decision Tree"]
classifiers_models = [model1, model2, rf, knn, naive_bayes, xgb_cl,dt]

# Create the empty dataframe
result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])


count = 0
for cls in classifiers_models:
    yproba = cls.predict(X_test1)
    fpr, tpr, _ = roc_curve(y_test1,  yproba)
    auc = roc_auc_score(y_test1, yproba)
    
    result_table = result_table.append({'classifiers':classifiers[count],
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc}, ignore_index=True)
    count = count+1
    
# Set name of the classifiers as index labels
result_table.set_index('classifiers', inplace=True)

fig = plt.figure(figsize=(8,6))
for i in result_table.index:
    plt.plot(result_table.loc[i]['fpr'], 
             result_table.loc[i]['tpr'], 
             label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
    
plt.plot([0,1], [0,1], color='orange', linestyle='--')

plt.xticks(np.arange(0.0, 1.1, step=0.1))
plt.xlabel("Flase Positive Rate", fontsize=15)

plt.yticks(np.arange(0.0, 1.1, step=0.1))
plt.ylabel("True Positive Rate", fontsize=15)

plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
plt.legend(prop={'size':13}, loc='lower right')

plt.savefig('performance results/statistic_features/All_models_Classification_Performances.png', transparent= False, bbox_inches= 'tight', dpi= 400)

plt.show()




#__________________________________________________________________________________________________________________


# # 3. Radial Kernel SVM
# # Just like the polynomial features method, the similarity features can be useful with any
# from sklearn.svm import SVC
# start_time3 = datetime.now()
# X_train3, X_test3, y_train3, y_test3 = train_test_split(X, y, test_size=0.2, random_state=42)
# model3 = SVC(kernel='rbf', gamma=0.5, C=0.1)
# model3.fit(X_train3, y_train3)
# print("3. Radial Kernel SVM")
# sk.sk_print_score(model3, X_train3, y_train3, X_test3, y_test3, train=True)
# sk.sk_print_score(model3, X_train3, y_train3, X_test3, y_test3, train=False)
# end_time3 = datetime.now()
# print('Duration: {}\n'.format(end_time3 - start_time3))

# from sklearn.svm import SVC
# from sklearn.decomposition import PCA
# pca = PCA(n_components=3)
# scaler = StandardScaler()
# X_train = pca.fit_transform(X_train)
# X_test = pca.transform(X_test)
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# plt.figure(figsize=(8,6))
# plt.scatter(X_train[:,0],X_train[:,1],c=y_train,cmap='plasma')
# plt.xlabel('First principal component')
# plt.ylabel('Second Principal Component')
# plt.show()

