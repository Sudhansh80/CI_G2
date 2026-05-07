import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

X=np.random.rand(100,1)
y=2*X + 1
model = LinearRegression()
model.fit(X, y)
joblib.dump(model, "src/model.pkl")

print("model trained successfully")