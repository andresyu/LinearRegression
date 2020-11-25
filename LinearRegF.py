from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io

df = pd.read_csv('lr.csv')
df = df.sample(frac = 1)
train_data,test_data=train_test_split(df,train_size=0.9,random_state=3)
y_test=np.array(test_data['y']).reshape(-1,1)

features = ['x1', 'x2', 'x3', 'x4']

reg=linear_model.LinearRegression()
reg.fit(train_data[features],train_data['y'])
pred=reg.predict(test_data[features])
mean_squared_error=metrics.mean_squared_error(y_test,pred)
print('MSE: ', round(np.sqrt(mean_squared_error),2))
print('R squared training: ',round(reg.score(train_data[features],train_data['y']),3))
print('R squared test: ', round(reg.score(test_data[features],test_data['y']),3))

plt.plot(range(len(y_test)), y_test, 'r.', label='Test')
plt.plot(range(len(y_test)), pred, 'b.', label='Predictions')
plt.legend()
plt.xlabel('# of instance')
plt.ylabel('Performance')
plt.show()

val = input("Enter your value: ")
while val != 'x':
 print('prediction ', pred[int(val)])
 print('real ', y_test[int(val)])
 val = input("Enter your value: ")
