import numpy as np
import pandas as pd
import optuna
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV, HalvingRandomSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import datetime
#from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
#from xgboost import XGBRegressor
#from catboost import CatBoostRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel



import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
import time
import datetime
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


# PATH
path='./_data/dacon_kcal/'
save_path= './_save/dacon_kcal/'

# CALL DATA
train_csv = pd.read_csv(path + 'train.csv', index_col = 0)
test_csv = pd.read_csv(path + 'test.csv', index_col = 0)
sample_submission_df = pd.read_csv(path + 'sample_submission.csv', encoding='utf-8')

# ID COLUMN REMOVE
col = ['ID',
        'Height(Feet)',
        'Height(Remainder_Inches)',
        'Weight(lb)',
        'Weight_Status',
        'Gender',
        'Age'
]

# ED, BT, BPM, CB
# CB, ED, BT, BPM


# Weight_Status, Gender => NUMBER
le_gender = LabelEncoder()
le_weight_status = LabelEncoder()

le_gender.fit(train_csv['Gender'])
le_weight_status.fit(train_csv['Weight_Status'])

train_csv['Gender'] = le_gender.transform(train_csv['Gender'])
train_csv['Weight_Status'] = le_weight_status.transform(train_csv['Weight_Status'])

test_csv['Gender'] = le_gender.transform(test_csv['Gender'])
test_csv['Weight_Status'] = le_weight_status.transform(test_csv['Weight_Status'])

train_csv['Total_Height_Inches'] = train_csv['Height(Feet)'] * 12 + train_csv['Height(Remainder_Inches)']
train_csv.drop(['Height(Feet)', 'Height(Remainder_Inches)'], axis=1, inplace=True)

test_csv['Total_Height_Inches'] = test_csv['Height(Feet)'] * 12 + test_csv['Height(Remainder_Inches)']
test_csv.drop(['Height(Feet)', 'Height(Remainder_Inches)'], axis=1, inplace=True)

x = train_csv.drop(['Calories_Burned'], axis = 1)
y = train_csv['Calories_Burned']

# PolynomialFeatures DATA 
poly = PolynomialFeatures(degree=2, include_bias=False)
X = poly.fit_transform(train_csv.drop('Calories_Burned', axis=1))
y = train_csv['Calories_Burned']

# SCALER
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# train, valid SPLIT
X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

n_splits = 50
kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 1234)

# parameters = [{'iterations':[1000,2000,1500],'learning_rate':[0.03,0.05,0.01]},{'depth':[6,5,2],'loss_function':['RMSE'],'task_type':['CPU']}]
parameters = [{"kernel":[ C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1)),
                          C(1.0, (1e-3, 1e3)) * RBF(20, (1e-2, 1e2)) + WhiteKernel(noise_level=1.2, noise_level_bounds=(1e-10, 1e+1)),
                          C(1.5, (1e-3, 1e3)) * RBF(15, (1e-2, 1e2)) + WhiteKernel(noise_level=1.5, noise_level_bounds=(1e-10, 1e+1)),],
               "n_restarts_optimizer": [5, 9, 12],"alpha": [1e-10, 1e-5]}]


model = RandomizedSearchCV(GaussianProcessRegressor(),
                     parameters,
                     cv = 20,
                     verbose = 1,
                     refit = True,
                     n_jobs = -1)

model.fit(X_train, y_train)

# valid PREDICT
y_pred_valid = model.predict(X_valid)
rmse_valid = np.sqrt(mean_squared_error(y_valid, y_pred_valid))
print(f"Valid 데이터 RMSE: {rmse_valid:.3f}")

# test PREDICT
X_test = test_csv.values
X_poly_test = poly.transform(X_test)
X_test_scaled = scaler.transform(X_poly_test)
y_pred_test = model.predict(X_test_scaled)

date = datetime.datetime.now()
date = date.strftime("%m%d-%H%M")
start_time = time.time()

# SUBMIT
sample_submission_df['Calories_Burned'] = y_pred_test
sample_submission_df.to_csv(save_path + date + 'submission_MLP_Poly.csv', index=False)
