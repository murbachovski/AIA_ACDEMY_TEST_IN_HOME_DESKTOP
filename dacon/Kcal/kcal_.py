import pandas as pd
import numpy as np
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict, HalvingRandomSearchCV, HalvingGridSearchCV
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler, LabelEncoder, QuantileTransformer
from sklearn.utils import all_estimators
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import all_estimators
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
import optuna
import datetime
import warnings
from sklearn.preprocessing import PolynomialFeatures
warnings.filterwarnings('ignore')


poly = PolynomialFeatures(degree=2, include_bias=False)

def RMSE(x, y):
    return np.sqrt(mean_squared_error(x, y))


# PATH
path='./_data/dacon_kcal/'
save_path= './_save/dacon_kcal/'
path_save_min = './_save/dacon_kcal/min/'

train_csv = pd.read_csv(path + 'train.csv')
test_csv = pd.read_csv(path + 'test.csv')
submit_csv = pd.read_csv(path + 'sample_submission.csv')

# Weight_Status, Gender => NUMBER
train_csv['Weight_Status'] = train_csv['Weight_Status'].map({'Normal Weight': 0, 'Overweight': 1, 'Obese': 2})
train_csv['Gender'] = train_csv['Gender'].map({'M': 0, 'F': 1})
test_csv['Weight_Status'] = test_csv['Weight_Status'].map({'Normal Weight': 0, 'Overweight': 1, 'Obese': 2})
test_csv['Gender'] = test_csv['Gender'].map({'M': 0, 'F': 1})

train_csv = train_csv.drop('ID', axis=1)
test_csv = test_csv.drop('ID', axis=1)

x = train_csv.drop('Calories_Burned', axis=1)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(train_csv.drop('Calories_Burned', axis=1))

poly = PolynomialFeatures(degree=2, include_bias=False)
x = poly.fit_transform(x_scaled)
y = train_csv['Calories_Burned']

min_rmse = 1
for k in range(1000000):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, shuffle=True, random_state=k)

    def objective(trial, x_train, y_train, x_test, y_test, min_rmse):
        alpha = trial.suggest_loguniform('alpha', 0.0001, 1)
        n_restarts_optimizer  = trial.suggest_int('n_restarts_optimizer', 3, 10)
        optimizer = trial.suggest_categorical('optimizer', ['fmin_l_bfgs_b'])

        model = GaussianProcessRegressor(
            alpha=alpha,
            n_restarts_optimizer=n_restarts_optimizer,
            optimizer=optimizer,
        )
        
        model.fit(x_train, y_train)
        
        print('GPR result : ', model.score(x_test, y_test))
        
        x_test_scaled = scaler.transform(test_csv)
        x_test = poly.transform(x_test_scaled)
        y_pred = model.predict(x_test)
        rmse = RMSE(y_test, y_pred)
        print('GPR RMSE : ', rmse)
        if rmse < 0.1:
            submit_csv['Calories_Burned'] = model.predict(test_csv)
            date = datetime.datetime.now()
            date = date.strftime('%m%d_%H%M%S')
            submit_csv.to_csv(save_path + date + str(round(rmse, 5)) + '.csv')
        return rmse
    
    opt = optuna.create_study(direction='minimize')
    opt.optimize(lambda trial: objective(trial, x_train, y_train, x_test, y_test, min_rmse), n_trials=20)
    print('best param : ', opt.best_params, 'best rmse : ', opt.best_value)