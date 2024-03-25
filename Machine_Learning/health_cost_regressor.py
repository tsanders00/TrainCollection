import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('insurance.csv', sep=',')

# encode cat values
label_encoder = LabelEncoder()
data.sex = label_encoder.fit_transform(data.sex)
data.smoker = label_encoder.fit_transform(data.smoker)
data.region = label_encoder.fit_transform(data.region)

# separate labels from data
labels = data.pop('expenses')

# split data
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# model
param_grid = {'n_estimators': [100, 200, 500], 'min_samples_leaf': [1, 2, 3, 4], 'max_depth': [2, 5, 10]}
regressor = RandomForestRegressor()
grid_search = GridSearchCV(regressor, param_grid=param_grid, scoring='neg_mean_absolute_error', verbose=3, n_jobs=-1)
grid_search.fit(x_train, y_train,)
predictions = grid_search.predict(x_test)
print(f'MAE should be below 3500, model achieved: {round(mean_absolute_error(y_test, predictions))}')
