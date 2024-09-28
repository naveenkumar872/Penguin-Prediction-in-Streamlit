# train_model.py
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


import joblib

# Load dataset
data=pd.read_csv("data/penguins-class.csv")

X=data[['Culmen Length (mm)','Culmen Depth (mm)','Flipper Length (mm)','Body Mass (g)']].copy()
#filling null values
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
X.iloc[:,0:3]=imputer.fit_transform(X.iloc[:,0:3])

y=data['Species'].copy()
#encoding categorical values
label_encoder=LabelEncoder()
y=label_encoder.fit_transform(y)
y

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'model.joblib')
