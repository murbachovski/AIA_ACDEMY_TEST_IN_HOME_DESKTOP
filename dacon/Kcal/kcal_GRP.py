import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
import time
import datetime
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import datetime
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

# PATH
path='./_data/dacon_kcal/'
save_path= './_save/dacon_kcal/'

# CALL DATA
train_df = pd.read_csv(path + 'train.csv')
test_df = pd.read_csv(path + 'test.csv')
sample_submission_df = pd.read_csv(path + 'sample_submission.csv')

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
train_df['Weight_Status'] = train_df['Weight_Status'].map({'Normal Weight': 0, 'Overweight': 1, 'Obese': 2})
train_df['Gender'] = train_df['Gender'].map({'M': 0, 'F': 1})
test_df['Weight_Status'] = test_df['Weight_Status'].map({'Normal Weight': 0, 'Overweight': 1, 'Obese': 2})
test_df['Gender'] = test_df['Gender'].map({'M': 0, 'F': 1})

train_df = train_df.drop('ID',  axis=1)
test_df = test_df.drop('ID', axis=1)

# PolynomialFeatures DATA 
poly = PolynomialFeatures(degree=2, include_bias=True)
X = poly.fit_transform(train_df.drop('Calories_Burned', axis=1))
y = train_df['Calories_Burned']

# SCALER
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# train, valid SPLIT
X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# model 13
#kernel = Matern()
#parameters = [{"kernel":[ C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1)),
#                          C(1.0, (1e-3, 1e3)) * RBF(20, (1e-2, 1e2)) + WhiteKernel(noise_level=1.2, noise_level_bounds=(1e-10, 1e+1)),
#                          C(1.5, (1e-3, 1e3)) * RBF(15, (1e-2, 1e2)) + WhiteKernel(noise_level=1.5, noise_level_bounds=(1e-10, 1e+1)),],
#               "n_restarts_optimizer": [5, 9, 12],"alpha": [1e-10, 1e-5]}]
kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10))
grp = GaussianProcessRegressor(
                        kernel=kernel,
                        alpha=1e-10,
                        optimizer='fmin_l_bfgs_b',
                        n_restarts_optimizer=1,
                        normalize_y=False,
                        copy_X_train=True,
                        random_state=None
                        )
grp.fit(X_train, y_train)

# valid PREDICT
y_pred_valid = grp.predict(X_valid)
rmse_valid = np.sqrt(mean_squared_error(y_valid, y_pred_valid))
print(f"Valid 데이터 RMSE: {rmse_valid:.3f}")

# test PREDICT
X_test = test_df.values
X_poly_test = poly.transform(X_test)
X_test_scaled = scaler.transform(X_poly_test)
y_pred_test = grp.predict(X_test_scaled)

date = datetime.datetime.now()
date = date.strftime("%m%d-%H%M")
start_time = time.time()

# SUBMIT
sample_submission_df['Calories_Burned'] = y_pred_test
sample_submission_df.to_csv(save_path + date + 'submission_grp_Poly.csv', index=False)


#kernel: 사용할 kernel function입니다. 기본값은 RBF입니다.
#alpha: kernel function의 variance를 scaling하기 위한 상수입니다. 기본값은 1e-10입니다.
#optimizer: hyperparameter를 최적화하기 위한 optimizer입니다. 기본값은 fmin_l_bfgs_b입니다.
#n_restarts_optimizer: optimizer가 실행될 횟수입니다. 기본값은 0입니다.
#normalize_y: 타겟 값의 normalization 여부입니다. 기본값은 False입니다.
#copy_X_train: 입력 데이터를 복사하여 사용할지 여부입니다. 기본값은 True입니다.
#random_state: 난수 생성기의 상태를 지정하는 값입니다. 기본값은 None입니다.