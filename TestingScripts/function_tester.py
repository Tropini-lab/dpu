import numpy as np
import pandas as pd
array1 = np.array([3,4,5,6,7,8,9,10])
array2 = ([2,5,1,3,5,7,4,12])

df = pd.DataFrame()

df['Array1'] = array1
df['Array2'] = array2

df.to_csv(r'C:\Users\erlyall\PycharmProjects\dpu\CarolinaTestData.csv', index = False)

print('Complete')