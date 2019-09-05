import os 
import numpy as np
import pandas as pd

fname = 'jena_climate_2009_2016.csv'
data = pd.read_csv(fname)
float_data = data[0]
print(float_data)