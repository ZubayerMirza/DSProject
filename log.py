import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
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

sns.lmplot(x= X_test, y= clf.predict_proba(X_test), logistic=True)