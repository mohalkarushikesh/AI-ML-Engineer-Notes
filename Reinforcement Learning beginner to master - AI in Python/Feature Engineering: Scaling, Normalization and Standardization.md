Feature engineering is process of creating, scaling and selecting most relevant variables (features) from the raw data to improve model performance 

various techniques 
	
	1. Absoulte maximum Scaling : rescales each feature by deviding all the values by maximum absolute value of that features, 
		it ensures features values fall between the range of -1 and 1
		it is highly sensitive to ouliers, which can skew the absolute value and negatively impact the scaling quantity 

		Xscaled = Xi / max(|X|)

import numpy as np 
import pandas as pd

df = pd.read_csv('Housing.csv')

df.select_datatypes(include = np.number)

df.head()

# absolute scaling

# find max absolute value
max_abs = np.max(np.abs(df), axis=0)

# take devide all values by max absolute value
scaled_df = df / max_abs

scaled_df.head()

	2. Min-max scaling : Transforms the features by subtracting the minimum value and deviding the difference between the maximum value and minimum values  
			The method maps the features values to specified range, commonly the 0 and 1
	
		X_scaled = X - Xmin / X_max - X_min 

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(df)

scaled_df = pd.DataFrame(scaled_data, columns = df.columns)
	
	2. Normalization : 


		

	3. Stadardization : 
