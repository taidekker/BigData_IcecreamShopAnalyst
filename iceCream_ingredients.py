#Napapis Dekker

import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_pacf

data = pd.read_csv('data/Ingredients.csv')

# Your code starts after this line

Flavor = data['Flavor']
Sales = data['Sales']

ing1 = data['Ingredient 1']
ing2 = data['Ingredient 2']
ing3 = data['Ingredient 3']
ing4 = data['Ingredient 4']
ing5 = data['Ingredient 5']
ing6 = data['Ingredient 6']
newFlavor = [47,31,48,18,49,43]

df = pd.DataFrame({'ing1':ing1,'ing2':ing2,'ing3':ing3,'ing4':ing4,'ing5':ing5,'ing6':ing6})
X = df
Y = Flavor
model = sm.OLS(Y,X)
print(model.fit().summary())

new_flavor = model.fit().predict(newFlavor)
print (round(float(new_flavor),2))


df_sale = pd.DataFrame({'ing1':ing1,'ing2':ing2,'ing4':ing4,'ing6':ing6})

X = df_sale
Y = Sales
X = sm.add_constant(X)
model = sm.OLS(Y,X)
print(model.fit().summary())

new_sales = [1,47,31,18,43] 
expected_sales = model.fit().predict(new_sales)
print (round(float(expected_sales),2))

# Your code ends before this line
