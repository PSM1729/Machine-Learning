import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset= pd.read_csv('Position_Salaries.csv')
x= dataset.iloc[:, 1:2].values
y= dataset.iloc[:, 2].values
