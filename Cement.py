import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor


data = pd.read_csv(r"C:\Users\Snehith\TDC\Projects\Kaggle\Concrete_Data_Yeh.csv")
print(data.head())
print(data.describe())
print(data.info())

    

sns.heatmap(data.corr(), cmap='Blues', annot=True)
#plt.show()
         
l = data.columns.values
number_of_columns=9
number_of_rows = len(l)-1/number_of_columns

plt.figure(figsize=(number_of_columns, number_of_rows))
for i in range(0,len(l)):
    sns.distplot(data[l[i]])
    #plt.show()

X = data.drop(['csMPa'], axis=1)
y = data.drop(['cement', 'slag', 'flyash', 'water', 'superplasticizer', 'coarseaggregate', 'fineaggregate', 'age'], axis=1)    

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#LinearRegression
linear = LinearRegression()
#print(linear.fit(X_train, y_train))
#print(linear.score(X_train, y_train))
#print(linear.predict(X_test))
#print(linear.score(X_test, y_test))
#plt.plot(y_train, linear.predict(X_train))
#plt.show()

#RandomForestRegression
Randomforest = RandomForestRegressor(n_estimators=1, random_state=10)
print(Randomforest.fit(X_train, y_train))
print(Randomforest.score(X_train, y_train))
print(Randomforest.predict(X_test))
print(Randomforest.score(X_test, y_test))
print(np.array(y_test, Randomforest.predict(X_test)))


