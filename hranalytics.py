import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

from sklearn import preprocessing

from imblearn.under_sampling import RandomUnderSampler

from sklearn.ensemble import RandomForestClassifier  

from imblearn.over_sampling import RandomOverSampler

from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_csv('/content/drive/MyDrive/Project/train_LZdllcl.csv')

df1 = pd.read_csv('/content/drive/MyDrive/Project/test_2umaH9m.csv')

"""## **EDA**"""

df.head()

df.tail()

df.info()

df.columns

plt.rcParams['figure.figsize'] = [10, 5]
ct = pd.crosstab(df.department,df.is_promoted,normalize='index')
ct.plot.bar(stacked=True)
plt.legend(title='is_promoted',bbox_to_anchor=(1,0.5))

"""**While Technology department had highest percentage of employees getting promoted, Legal department has the least number. But we don't see major differences in terms of percentages.**"""

reg = pd.crosstab(df.region,df.is_promoted,normalize='index')
reg.plot.bar(stacked=True)
plt.legend(title='is_promoted',bbox_to_anchor=(1,0.5))

plt.rcParams['figure.figsize'] = [5, 5]
edu = pd.crosstab(df.education,df.is_promoted,normalize='index')
edu.plot.bar(stacked=True)
plt.rcParams['figure.figsize'] = [5, 5]
plt.legend(title='is_promoted',bbox_to_anchor=(1,0.5))

"""**As we can see the percentages are pretty much the same aross different educational backgrounds.**"""

pd.crosstab(df.gender,df.is_promoted,normalize='index')

pd.crosstab(df.recruitment_channel,df.is_promoted,normalize='index')

"""**According to the data, percentage of promotions is higher among the employees who got recruited through referrals.**"""

pd.crosstab(df['KPIs_met >80%'],df.is_promoted,normalize='index')

"""**According to the data, percentage of promotions is higher among the employees who got recruited through referrals.**"""

rating = pd.crosstab(df.previous_year_rating,df.is_promoted,normalize='index')
rating.plot.bar(stacked=True)
plt.legend(title='is_promoted',loc='upper left',bbox_to_anchor=(1, 0.5))

"""**The ratio of promoted employees increases with previous year rating which is quite obvious.**"""

df.corr(method="pearson")

"""## **Feature** **Engineering**

Playing dummies with multiple columns

First department ->
"""

df['department'].value_counts()

dummies = pd.get_dummies(df.department)

dummies

df = pd.concat([df,dummies],axis = 'columns')

df

df = df.drop(['department','R&D'], axis='columns')

df.info()

"""Now gender ->"""

dummies = pd.get_dummies(df.gender)

dummies

df = pd.concat([df,dummies],axis = 'columns')

df = df.drop(['gender', 'f'], axis='columns')

df.info()

"""**Now Education ->**"""

dummies = pd.get_dummies(df.education)

dummies

df = pd.concat([df,dummies],axis = 'columns')

df = df.drop(['education', 'Below Secondary'], axis='columns')

df.info()

"""**Now recruitment channel ->**"""

dummies = pd.get_dummies(df.recruitment_channel)

dummies

df = pd.concat([df,dummies],axis = 'columns')

df = df.drop(['recruitment_channel'], axis='columns')

df.info()

df

"""**Analysing region again for decision making ->**"""

df.region.unique()

fig, ax = plt.subplots(figsize=(35, 15))

br = sns.barplot(x='region', y='is_promoted', data=df)

plt.show()

df = df.drop(['region'],axis='columns')

df.info()

# dummies = pd.get_dummies(df.region)

# dummies

# df = pd.concat([df,dummies],axis = 'columns')

# df = df.drop(['region', 'region_1'], axis='columns')

df = df.drop(['employee_id'],axis='columns')

df.info()

# df = df.drop(['age'],axis='columns')

"""## **Data Preprocessing**"""

df.isnull().sum()

mean_value = df['previous_year_rating'].mean()
df['previous_year_rating'].fillna(value=mean_value, inplace=True)

df.isnull().sum()

df.info()

"""# **Analyzation based on Heat map**"""

plt.figure(figsize=(20,18))
sns.heatmap(df.corr(),annot = True, cmap="Accent")

"""# **Model Development**"""

y_start = df['is_promoted']
x_start =  df.drop(['is_promoted'],axis='columns')

x_start.info()

"""# **Model Tuning**"""

# rus = RandomUnderSampler(random_state=0)
# x, y = rus.fit_resample(x_start, y_start)

ros = RandomOverSampler(random_state=0)
x, y = ros.fit_resample(x_start, y_start)

x.info()

# scaler = preprocessing.StandardScaler().fit(x)
# scaler

# # x = scaler.transform(x)
# x

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

"""# **Model Selection and Finalization**"""

classifier1 = LogisticRegression()
classifier1.fit(x_train, y_train)

predicted_y = classifier1.predict(x_test)
predicted_y

confusion_matrix(y_test,predicted_y)

precision_score(y_test,predicted_y)

f1_score(y_test,predicted_y)

recall_score(y_test,predicted_y)

gbr = GradientBoostingClassifier(n_estimators = 100)
  
# Fit to training set
gbr.fit(x_train, y_train)
  
# Predict on test set
predicted_y = gbr.predict(x_test)

confusion_matrix(y_test,predicted_y)

precision_score(y_test,predicted_y)

recall_score(y_test,predicted_y)

f1_score(y_test,predicted_y)

"""# **Final Model -> Random Forest**"""

classifier= RandomForestClassifier(n_estimators= 50, criterion="entropy")  
classifier.fit(x_train, y_train)

predicted_y = classifier.predict(x_test)
predicted_y

"""# **Model Performance / Accuracy**"""

confusion_matrix(y_test,predicted_y)

precision_score(y_test,predicted_y)

recall_score(y_test,predicted_y)

"""# **Final result that matters** **(F1 score)**"""

f1_score(y_test,predicted_y)

"""# **Making submission file for a test dataset ->**"""

# dummies = pd.get_dummies(df1.department)
# df1 = pd.concat([df1,dummies],axis = 'columns')
# df1 = df1.drop(['department','R&D'], axis='columns')
# dummies = pd.get_dummies(df1.gender)
# df1 = pd.concat([df1,dummies],axis = 'columns')
# df1 = df1.drop(['gender', 'f'], axis='columns')
# dummies = pd.get_dummies(df1.education)
# df1 = pd.concat([df1,dummies],axis = 'columns')
# df1 = df1.drop(['education', 'Below Secondary'], axis='columns')
# dummies = pd.get_dummies(df1.recruitment_channel)
# df1 = pd.concat([df1,dummies],axis = 'columns')
# df1 = df1.drop(['recruitment_channel'], axis='columns')
# df1 = df1.drop(['region'],axis='columns')
# df1 = df1.drop(['employee_id'],axis='columns')

# df1.info()

#  df1.isnull().sum()

# mean_value = df1['previous_year_rating'].mean()
# df1['previous_year_rating'].fillna(value=mean_value, inplace=True)

# x_test =  df1

# predicted_y = classifier.predict(x_test)
# predicted_y

# df1 = pd.read_csv('/content/drive/MyDrive/Project/test_2umaH9m.csv')

# df2 = pd.DataFrame(predicted_y, columns=['is_promoted'])
# df3 = pd.concat([df1['employee_id'],df2],axis = 'columns')

# df3.info()

# df3

# df3.to_csv('/content/drive/MyDrive/Project/submission.csv',index = False)
