import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model 
from sklearn.metrics import  mean_squared_error
dibetese =  datasets.load_diabetes()

#dibetese_x= dibetese.data[:, np.newaxis, 2]
dibetese_x= dibetese.data


dibetese_x_train = dibetese_x[:-30]
dibetese_x_test = dibetese_x[-30:]


dibetese_y_train = dibetese.target[:-30]
dibetese_y_test = dibetese.target[-30:]

model = linear_model.LinearRegression()

model.fit(dibetese_x_train,dibetese_y_train)

dibetese_y_pridict=model.predict(dibetese_x_test)

print("Mean squared error is: ", mean_squared_error(dibetese_y_test, dibetese_y_pridict))

print("weights: ", model.coef_)
print("intercept: ", model.intercept_)

# plt.scatter(dibetese_x_test,dibetese_y_test)
# plt.plot(dibetese_x_test,dibetese_y_pridict)
# plt.show()