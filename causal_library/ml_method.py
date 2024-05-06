
import warnings
warnings.filterwarnings('ignore')
import econml
from econml.dml import CausalForestDML
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from datetime import datetime
from joblib import Parallel, delayed
from econml.metalearners import TLearner, SLearner, XLearner, DomainAdaptationLearner
from sklearn.linear_model import LinearRegression,LassoCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor,RandomForestClassifier
from econml.dr import DRLearner
from sklearn import clone
from  lightgbm import LGBMRegressor, LGBMClassifier
import xgboost as xgb

class CustomLogisticRegressionCV:
    def __init__(self,name="CustomLogisticRegressionCV"):
        self.model = sklearn.linear_model.LogisticRegressionCV(max_iter=400, n_jobs=10)
        self.name = name
    def fit(self,X,y):
        self.model.fit(X,y)
    def predict(self,X):
        return self.model.predict_proba(X)[:,1]
    def get_params(self, deep=True):
        return {"name": self.name}
    def __sklearn_clone__(self):
        return self

class CustomForestClassifier:
    def __init__(self,name="CustomForestClassifier"):
        # self.model = RandomForestClassifier(n_estimators=200, max_depth=12,min_samples_leaf=200, random_state=0,n_jobs=40, verbose=1)
        self.model = LGBMRegressor(boosting_type='gbdt', max_depth=10, n_estimators=200, n_jobs=40, min_child_samples=100)
        self.name = name
    def fit(self,X,y):
        self.model.fit(X,y)
    def predict(self,X):
        return self.model.predict_proba(X)[:,1]
    def get_params(self, deep=True):
        return {"name": self.name}
    def __sklearn_clone__(self):
        return self
    

def getDML(model_param):
    param_n_estimators = model_param["n_estimators"] if "n_estimators" in model_param else 100
    param_max_depth = model_param["param_max_depth"] if "param_max_depth" in model_param else 8 
    param_min_samples_leaf = model_param["param_min_samples_leaf"] if "param_min_samples_leaf" in model_param else 2000
    param_criterion = model_param["criterion"] if "criterion" in model_param else 'mse'
    param_binaryY = model_param["y_binary"] if "y_binary" in model_param else False
    n_jobs = model_param["jobs"] if "jobs" in model_param else 20
    
    
    if(param_binaryY):
        print("Binary DML")
        model = CausalForestDML(
                        model_y= CustomLogisticRegressionCV(),
                        criterion= param_criterion , n_estimators= param_n_estimators,     
                        min_samples_leaf=param_min_samples_leaf,
                        max_depth= param_max_depth, discrete_treatment=True , n_jobs=n_jobs)
    else:
        model = CausalForestDML(
                criterion= param_criterion , n_estimators= param_n_estimators,     
                min_samples_leaf=param_min_samples_leaf,
                max_depth= param_max_depth, discrete_treatment=True , n_jobs=n_jobs)
    return model


def select_param_with_model_parallel(data, test_data, param):
    Y_sample, T_sample,X_sample = data
    model_param = {"param_max_depth": param[0], "param_min_samples_leaf":param[1], "criterion":param[2],"n_estimators":param[3] }

    model = getDML(model_param)
    model.fit(Y_sample, T_sample, X=X_sample)
    uplift = model.effect(X_sample)
    _,_,auuc = evaluate_auuc_with_data(uplift, T_sample, Y_sample)

    Y_test_sample, T_test_sample,X_test_sample = test_data
    uplift_test = model.effect(X_test_sample)
    _,_,auuc_test = evaluate_auuc_with_data(uplift_test, T_test_sample, Y_test_sample)

    return [model_param, auuc, auuc_test]


def modelDmlTrainAndPredict(model_param, train_data, test_data, X_valid_list, cv_param={"use_cv":False}):
    Y,T,X = train_data
    Y_test, T_test, X_test = test_data

    if len(np.unique(Y.flatten()).tolist()) == 2:
        print(np.unique(Y.flatten()))
        model_param["y_binary"]= True 
    print("Current Time1 =", datetime.now().strftime("%H:%M:%S"))

    if("use_cv" in cv_param):
        #use part of data to grid search
        sample_length = len(Y)
        sample_cv_frac =  cv_param["cv_frac"] if "cv_frac" in cv_param else 0.3
        Y_sample = Y[0:int(sample_cv_frac * sample_length)]
        T_sample = T[0:int(sample_cv_frac * sample_length)]
        X_sample = X[0:int(sample_cv_frac * sample_length)]

        model_param_cand = cv_param["param_cand"]
        data = (Y_sample, T_sample,X_sample)
        Y_test_sample = Y_test[0:int(0.4 * len(Y_test))]
        T_test_sample = T_test[0:int(0.4 * len(Y_test))]
        X_test_sample = X_test[0:int(0.4 * len(Y_test))]
        test_data = (Y_test_sample, T_test_sample,X_test_sample)
        cv_param_list = Parallel(n_jobs= -1)(delayed(select_param_with_model_parallel)(data, test_data, param) for param in model_param_cand)
        
        print(cv_param_list)
        print("Current Time2 =", datetime.now().strftime("%H:%M:%S"))
        best_param = max(cv_param_list, key = lambda row: row[2])
        model_param = best_param[0]
        model_param["n_estimators"] = 400

        model = getDML(model_param)
        model.fit(Y, T, X=X)
        return model, model.effect(X), model.effect(X_test), [model.effect(X_valid) for X_valid in X_valid_list]

    else:
        model = getDML(model_param)
        model.fit(Y, T, X=X)
        return model, model.effect(X), model.effect(X_test), [model.effect(X_valid) for X_valid in X_valid_list]


def modelTNTrainAndPredict(model_param, train_data, test_data, X_valid_list, cv_param={"use_cv":False}):
    Y,T,X = train_data
    Y_test, T_test, X_test = test_data
    n_jobs = model_param["jobs"] if "jobs" in model_param else 20
    print("Current Time1 =", datetime.now().strftime("%H:%M:%S"))
    models = RandomForestRegressor(max_depth=12,min_samples_leaf=200, random_state=0,n_jobs=n_jobs, verbose=1)
    if len(np.unique(Y.flatten()).tolist()) == 2:
        print(np.unique(Y.flatten()))
        models = CustomForestClassifier()
    model = TLearner(models=models)
    model.fit(Y, T, X=X)
    return model, model.effect(X), model.effect(X_test), [model.effect(X_valid) for X_valid in X_valid_list]

def modelTNTrainAndPredictCustom(model_param, train_data, test_data, X_valid_list, cv_param={"use_cv":False}):
    Y,T,X = train_data
    Y, T, X = check_inputs(Y, T, X)
    Y_test, T_test, X_test = test_data
    Y_test, T_test, X_test = check_inputs(Y_test, T_test, X_test)
    n_jobs = model_param["jobs"] if "jobs" in model_param else 20
    print("Current Time1 =", datetime.now().strftime("%H:%M:%S"))
    # models = RandomForestRegressor(max_depth=12,min_samples_leaf=200, random_state=0,n_jobs=n_jobs, verbose=1)
    models = LGBMRegressor(boosting_type='gbdt', max_depth=10, n_estimators=200, n_jobs=40, min_child_samples=100)
    is_binary = False
    if len(np.unique(Y.flatten()).tolist()) == 2:
        print(np.unique(Y.flatten()))
        models = LGBMClassifier(boosting_type='gbdt', max_depth=10, n_estimators=200, n_jobs=40, min_child_samples=100)
        is_binary = True
    y_model = []
    for i in range(2):
        model = clone(models)
        model.fit(X[T==i], Y[T==i])
        y_model.append(model)
    
    def effect(y_model, X):
        if is_binary:
            return y_model[1].predict_proba(X)[:,1] - y_model[0].predict_proba(X)[:,1]    
        else:
            return y_model[1].predict(X)[:,1] - y_model[0].predict(X)[:,1] 
    return model, effect(y_model,X), effect(y_model,X_test), [effect(y_model,X_valid) for X_valid in X_valid_list]

def modelXNTrainAndPredict(model_param, train_data, test_data, X_valid_list, cv_param={"use_cv":False}):
    Y,T,X = train_data
    Y_test, T_test, X_test = test_data

    n_jobs = model_param["jobs"] if "jobs" in model_param else 20
    print("Current Time1 =", datetime.now().strftime("%H:%M:%S"))
    if("model_dict" in model_param): # if model ready
        print("model is ready")
        model = model_param["model_dict"][model_param["model_name"]]
        return model, model.effect(X), model.effect(X_test), [model.effect(X_valid) for X_valid in X_valid_list]
    
    models = LGBMRegressor(boosting_type='gbdt', max_depth=10, n_estimators=200, n_jobs=n_jobs, min_child_samples=100)
    if len(np.unique(Y.flatten()).tolist()) == 2:
        print(np.unique(Y.flatten()))
        models = CustomForestClassifier()
    cate_models= LGBMRegressor(boosting_type='gbdt', max_depth=8, n_estimators=200, n_jobs=n_jobs, min_child_samples=100)

    model = XLearner(models = models,cate_models = cate_models,propensity_model=LGBMClassifier(boosting_type='gbdt', max_depth=10, n_estimators=200, n_jobs=40, min_child_samples=100))
    model.fit(Y, T, X=X)
    return model, model.effect(X), model.effect(X_test), [model.effect(X_valid) for X_valid in X_valid_list]