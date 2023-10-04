from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

sonar_data = pd.read_csv('sonar data.csv', header=None)



y = sonar_data[60]
X= sonar_data.drop(columns=60, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

model = LogisticRegression()
model.fit(X_train, y_train)

#Prediction accuracy on train data
X_train_predict = model.predict(X_train)
print('Training data accuracy score: {}'.format(accuracy_score(X_train_predict, y_train)))

#Prediction accuracy on test data
X_test_predict = model.predict(X_test)
print('Test data accuracy score: {}'.format(accuracy_score(X_test_predict, y_test)))

#predicting random data:
#It is a Rock
trial = np.asarray([0.0408,0.0653,0.0397,0.0604,0.0496,0.1817,0.1178,0.1024,0.0583,0.2176,0.2459,0.3332,0.3087,0.2613,0.3232,0.3731,0.4203,0.5364,0.7062,0.8196,0.8835,0.8299,0.7609,0.7605,0.8367,0.8905,0.7652,0.5897,0.3037,0.0823,0.2787,0.7241,0.8032,0.8050,0.7676,0.7468,0.6253,0.1730,0.2916,0.5003,0.5220,0.4824,0.4004,0.3877,0.1651,0.0442,0.0663,0.0418,0.0475,0.0235,0.0066,0.0062,0.0129,0.0184,0.0069,0.0198,0.0199,0.0102,0.0070,0.0055])
print(model.predict(trial.reshape(1,-1)))