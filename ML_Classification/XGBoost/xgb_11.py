from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
import xgboost as xgb

df = pd.read_csv('Data/ames_unprocessed_data.csv')
# Fill missing values with 0
X, y = df.iloc[:, :20], df.iloc[:, -1]
X.LotFrontage = X.LotFrontage.fillna(0)

steps = [("ohe_onestep", DictVectorizer(sparse=False)),
         ("xgb_model", xgb.XGBRegressor(max_depth=2, objective="reg:linear"))]

xgb_pipeline = Pipeline(steps)
cross_val_scores = cross_val_score(estimator=xgb_pipeline, X=X.to_dict("records"),
                                   y=y, scoring="neg_mean_squared_error", cv=10)
# xgb_pipeline.fit(X.to_dict("records"), y)
print("10-fold RMSE: ", np.mean(np.sqrt(np.abs(cross_val_scores))))
print()