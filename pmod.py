import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from scipy.special import expit
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

chs = pd.read_sas('chs2020_public.sas7bdat')
chs['Mental Health'] = (chs['mhtreat20_all'] == 1) | (chs['nspd'] == 1) | (chs['mood11'] == 1)
X = chs['imputed_povertygroup'] # Independent Variable
y = chs['Mental Health'] # Dependent Variable to predict

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train = X_train.values.reshape(-1,1)
X_test = X_test.values.reshape(-1,1)

clf = LogisticRegression(C=.9)
clf.fit(X_train, y_train.values)

def jitter_df(df, x_col, y_col):
    x_jittered = df[x_col] + np.random.normal(scale=0.3, size=len(df))
    y_jittered = df[y_col] + np.random.normal(scale=0.02, size=len(df))
    return df.assign(**{x_col: x_jittered, y_col: y_jittered})

# Create the graph

sns.regplot('imputed_povertygroup', "Mental Health", data = jitter_df(chs, 'imputed_povertygroup', 'Mental Health'), fit_reg=False, logistic=True)

#This will be the X values used to predict
xs = X_test
#Predict the Y's
ys = expit(xs * clf.coef_ + clf.intercept_)
# ys = clf.predict_proba(xs.reshape(-1, 1))[:, 1]

xs, ys = xs.reshape(-1,1), ys.reshape(-1,1)
plt.plot(xs,ys, color = 'r', linewidth= 3)
# loss = expit(X_test.values * clf.coef_ + clf.intercept_)
# plt.plot(X_test.values.reshape(1,-1), loss, color="red", linewidth=3)

plt.ylabel("Mental Health")
plt.yticks([0,1], ['Normal', 'Low'])
plt.xticks(np.arange(1,6), ['<100%', '100-200%', '200-400%', '400-600%', '600%<'])
plt.xlabel('Poverty Group')
plt.title('Logistic Regression of Poverty Level')

# # Linear Regression

# ols = LinearRegression()

# # plt.legend(
# #     ("Logistic Regression Model", "Linear Regression Model"),
# #     loc="lower right",
# #     fontsize="small",
# # )
# plt.tight_layout()
# plt.savefig('graph/predictive_model/lr.png')
plt.show()

