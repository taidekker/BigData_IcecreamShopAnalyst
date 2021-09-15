import pandas as pd
import statistics as st
import numpy as np
from scipy import stats

data = pd.read_csv("data/ages.csv")

# Your code starts after this line
"""
The file ages.csv contains the ages of 100 random customers that came to the store during the past year.
Assuming the ages are normally distributed, 
calculate the probability that a person older than 40 will come to the store. Print the result rounded to two decimals
"""
age = data["Age"]

 
age_mean = np.mean(age)
age_sd = np.std(age)

max_age = max(age)

#default normal distribution is mean = 0, sd = 1
p = stats.norm.cdf(max_age, loc = age_mean, scale = age_sd)-stats.norm.cdf(40, loc = age_mean, scale = age_sd)
print( round(p, 2))

# Your code ends before this line