import xgboost as xgb
from sklearn import linear_model
from sklearn.base import BaseEstimator

class Regressor(BaseEstimator):
    def __init__(self, params=None):
        self.clf = xgb.sklearn.XGBRegressor(max_depth=3,
                                            learning_rate=0.1,
                                            n_estimators=300,
                                            silent=True,
                                            objective='reg:linear',
                                            nthread=1,
                                            gamma=0,
                                            min_child_weight=1,
                                            max_delta_step=0,
                                            subsample=1,
                                            colsample_bytree=1,
                                            colsample_bylevel=.25, #.5
                                            reg_alpha=0, #1
                                            reg_lambda=.5, #.2
                                            scale_pos_weight=1,
                                            base_score=0.5,
                                            seed=0,
                                            missing=None)
            
        self.clf2 = linear_model.LassoLarsCV(fit_intercept=True)
    
    def fit(self, X, y):
        self.clf.fit(X,y)
        self.clf2.fit(X,y)
    
    def predict(self, X):
        tmp = self.clf.predict(X)
        tmp2= self.clf2.predict(X)
        return (tmp+tmp2)/2
        