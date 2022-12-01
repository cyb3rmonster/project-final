import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import seaborn as sns

df_icu = pd.read_csv('data01.csv')

df_icu = df_icu.dropna(subset=['outcome'])
y = df_icu.iloc[:,2:3]
X = df_icu.iloc[:,3:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 2021)

ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)
y_train = np.array(y_train)

rfr = RandomForestRegressor()
rfr.fit(X_train_scaled, y_train)
display(rfr.score(X_train_scaled, y_train))

feats = {}
for feature, importance in zip(df_icu.columns, rfr.feature_importances_):
    feats[feature] = importance
importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-Importance'})
importances = importances.sort_values(by='Gini-Importance', ascending=False)
importances = importances.reset_index()
importances = importances.rename(columns={'index': 'Features'})
sns.set(font_scale = 5)
sns.set(style="whitegrid", color_codes=True, font_scale = 1.7)
fig, ax = plt.subplots()
fig.set_size_inches(30,15)
sns.barplot(x=importances['Gini-Importance'], y=importances['Features'], data=importances, color='skyblue')
plt.xlabel('Importance', fontsize=25, weight = 'bold')
plt.ylabel('Features', fontsize=25, weight = 'bold')
plt.title('Feature Importance', fontsize=25, weight = 'bold')
display(plt.show())
display(importances)