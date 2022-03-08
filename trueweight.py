import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as logistic_model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def sigmoid(x):
	return 1/(1+np.exp(-x))

upweight_list = {}
non_upweight_list = {}
n=10000
d=10
w_true = np.random.rand(d,1)
x = np.random.normal(0,1,(n,d))
for experiment in range(1000):
	y_logit = x @ w_true + np.random.normal(0,1,(n,1))
	y_prob = sigmoid(y_logit.squeeze())
	y_label = y_prob>(experiment/1000)

	imbalance = (y_label.sum()/len(y_label))
	if(imbalance == 0 or imbalance == 1):
		continue

	upweight = logistic_model(class_weight='balanced').fit(x,y_label)
	non_upweight = logistic_model().fit(x,y_label)

	upweight_mse = mean_squared_error(w_true.ravel(),upweight.coef_.ravel())
	upweight_list[imbalance] = upweight_mse
	non_upweight_mse = mean_squared_error(w_true.ravel(), non_upweight.coef_.ravel())
	non_upweight_list[imbalance] = non_upweight_mse
x,y = zip(*sorted(non_upweight_list.items()))
df = pd.DataFrame.from_dict(non_upweight_list, orient='index',columns = ['non_upweight_mse'])
df['upweight_mse'] = upweight_list
print(df)

plt.plot(x,y, label='Non Upweighted MSE')
x,y = zip(*sorted(upweight_list.items()))
plt.plot(x,y, label='Upweighted MSE')
plt.legend()
plt.title('MSE of Logistic Model Weights vs. True Weights')
plt.xlabel('Imbalance (positive class / total)')
plt.ylabel('MSE')
plt.savefig('weights.png')
