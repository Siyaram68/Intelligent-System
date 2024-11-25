import pandas as pd
s= pd.Series([3,-5,7,4],index=['a','b','c','d'])
print (s)
s= pd.Series([3,-5,7,4])
print (s)
data=[['Aniket',100],['Siyaram',150],['Indraneel',50]]
df= pd.DataFrame(data,columns=['Name','Price'])
print(df)

ds1=df.sort_index()
print(ds1)
data=[['Aniket',100],['Siyaram',150],['Indraneel',50]]
df= pd.DataFrame(data)
print(df)

ds1=df.sort_index()
print(ds1)

x=(df.rank())
print(x)

type(df)
pd.read_csv("ds1.csv")
from matplotlib import pyplot as plt
x=[1,2,3,4,5]
y=[2,4,8,1,6]
plt.plot(x,y)
plt.show()
plt.bar(x,y)
plt.show()
plt.hist(x)
plt.show()
plt.hist(y)
plt.show()
plt.scatter(x,y)
plt.show()
import matplotlib.pyplot as plt
import numpy as np
z=np.array([100,40,50,20,70])
mylabels=["Siyaram","Amrit","Indraneel","Priyank","Zeeshan"]
plt.pie(z,labels=mylabels)
plt.show()
import matplotlib.pyplot as plt
month=["Jan","Feb","Mar","Apr","May"]
India=[100.6,134.67,123.2,145.34,165.7]
BD=[23.1,10.3,46.3,145.7,110.5]
plt.plot(month,India,color="orange",label="India")
plt.plot(month,BD,color="green",label="Bangladesh")
plt.show()
