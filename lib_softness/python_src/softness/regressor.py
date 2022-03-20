import sys, os
import numpy as np
import numpy.ma as ma
import scipy as sp
import pandas as pd
import sklearn as skl
import sklearn.linear_model as lin
import sklearn.preprocessing as prep
import sklearn.pipeline as pipeline
import sklearn.model_selection as modsel
import sklearn.metrics as metrics
import sklearn.feature_selection as featsel
import sklearn.ensemble as ensemble



def get_regressor(df, x_col, y_col, random_state=777, reg_type='RIDGE', kwargs=dict()):

    X = df[x_col].values
    Y = df[y_col].values.ravel()


    if reg_type == 'RIDGE':
#         reg = pipeline.make_pipeline(prep.StandardScaler(),
#                                  lin.Ridge(random_state=random_state, **kwargs))

        reg = pipeline.make_pipeline(prep.StandardScaler(),
                                 lin.Ridge(random_state=random_state, **kwargs))
  
    reg.fit(X, Y)

    print("score:", reg.score(X, Y))

    return reg