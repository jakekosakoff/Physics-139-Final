import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
random_state = 3

Boosting_data = pd.read_csv("Data/BoostingData.csv")

X = Boosting_data.values[:,3:]
y = Boosting_data.values[:,2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random_state)
X_train = np.asarray(X_train.astype("float32"))
X_test = np.asarray(X_test.astype("float32"))



from xgboost import XGBRegressor



bdt = XGBRegressor(n_estimators=100,max_depth=3,learning_rate=0.1,colsample_bytree=0.8, subsample=0.8)
print(bdt)
bdt.fit(X_train, y_train)
preds_bdt = bdt.predict(X_test)


import xgboost as xgb
#xgb.plot_importance(bdt)
plt.scatter(preds_bdt, y_test)
plt.plot(np.arange(200, 400), np.arange(200,400), color= "r")
plt.xlabel('Predicted energy values') 
plt.ylabel('Actual energy values')
plt.legend()
plt.title("Predicted vs Actual energy values")
plt.savefig("Plots/predicted_actual_energy")
plt.show()

'''
#count Signal and BAckground
cnt_signal=0
cnt_back=0
for i in range(len(y_test)):
    if (y_test[i]==0):
        cnt_back=cnt_back+1
    else:
        cnt_signal=cnt_signal+1
#print(cnt_signal)
#print(cnt_back)
#Filter Signal and background
sig=np.zeros((cnt_signal))
back=np.zeros((cnt_back))
cnt_signal=0
cnt_back=0
f_12=12  
f_13=13
#this fearyre will have gaus distribution 
#f_11=11  
#f_12=12



for i in range(len(y_test)):
    if (y_test[i]==0):
        back[cnt_back]=X_test[i,f_12:f_13]
        cnt_back=cnt_back+1
    else:
        sig[cnt_signal]=X_test[i,f_12:f_13]
        cnt_signal=cnt_signal+1
    
    

#signal  it has gous distribution 
counts, bins = np.histogram(sig,bins=100)
#plt.stairs(counts, bins)
#plt.show()

plt.hist(bins[:-1], bins, weights=counts)
plt.show()

#backgroung
counts, bins = np.histogram(back,bins=100)
plt.hist(bins[:-1], bins, weights=counts)
plt.show()'''

