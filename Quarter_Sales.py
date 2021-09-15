import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_pacf

data = pd.read_csv('data/Quarter_Sales.csv')

# Your code starts after this line
plot_pacf(data['Sales'])

sale = 'Sales'
shift = [1,3,4,5,6,8,11,18]
Sale = data[sale]

s1 = Sale.shift(periods = 1)
s3 = Sale.shift(periods = 3)
s4 = Sale.shift(periods = 4)
s5 = Sale.shift(periods = 5)
s6 = Sale.shift(periods = 6)
s8 = Sale.shift(periods = 8)
s11 = Sale.shift(periods = 11)
s18 = Sale.shift(periods = 18)

predictor = pd.DataFrame({'s4':s4,'s6':s6,'s18':s18})
Y = Sale[18:]
X = predictor[18:]

model = sm.OLS(Y,X)
print(model.fit().summary())

t = Sale.size
quarter = pd.DataFrame({'s4':[Sale[t-4]],'s6':[Sale[t-6]],'s18':[Sale[t-18]]})
q = model.fit().predict(quarter)
print ( round(q,2))
# Your code ends before this line