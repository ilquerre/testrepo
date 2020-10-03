# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
data=pd.read_csv("positions.csv")

level=data.iloc[:,1].values.reshape(-1,1)
salary=data.iloc[:,2].values.reshape(-1,1)

regressor=LinearRegression()

# regressor.fit(level,salary)

x=np.arange(min(data.Level),int(max(data.Level)+1)).reshape(-1,1)

# plt.scatter(data.Level,data.Salary)
# plt.xlabel("Level of Position")
# plt.ylabel("Salary")
# plt.title("Linear Regression for Level-Salary(not true)")
# gördüğümüz gibi bu model uygun değil
# çünü maaş ve level arasında tam lineer korelasyon yok düz çizgi olmaz
# level arttıkça maaş artıyor ama sabit ve orantılı bir artış değil
# POLİNOMİK BİR ARTIŞ

RegressorPoly=PolynomialFeatures(degree=4) #derece değişebilir
levelPoly=RegressorPoly.fit_transform(level)
# polinom iiçin preprocessing
regressor.fit(levelPoly,salary)
value=np.array(input("Level?")).astype(float)
print(regressor.predict(RegressorPoly.fit_transform([[value]])))

plt.scatter(data.Level,data.Salary)
plt.xlabel("Level of Position")
plt.ylabel("Salary")
plt.plot(x,regressor.predict(RegressorPoly.fit_transform(x)))

skor=r2_score(salary,regressor.predict(levelPoly))
