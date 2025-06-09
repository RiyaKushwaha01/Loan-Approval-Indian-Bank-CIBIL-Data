#!/usr/bin/env python
# coding: utf-8

# ### 1. Import Libraries

# In[1]:


import pyodbc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.feature_selection import RFE, RFECV, SelectKBest, f_classif
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support


# ### 2. Load the Dataset

# ##### Connect the SQL Server to get the data

# In[5]:


# Eshtablishing the connection :

conn = pyodbc.connect(
    r'DRIVER={ODBC Driver 17 for SQL Server};'
    r'SERVER=DESKTOP-DDTGTVD\SQLEXPRESS02;'
    r'DATABASE=BANK_DB;'
    r'Trusted_Connection=yes;'
)


# In[7]:


# If connection is succesfull, print True
if conn:
    print('True')


# In[77]:


df1 = "SELECT * FROM Internal_Bank_Data "
df2 = "SELECT * FROM External_Data"


# In[79]:


df1 = pd.read_sql(df1, conn)
df2 = pd.read_sql(df2, conn)


# In[13]:


df1.head(2)


# In[15]:


df2.head(2)


# ### 3. Data Exploration :

# ##### For df1

# In[81]:


# Basic informationabout the dataset:
df1.info()


# In[83]:


# Summary 
df1.describe()


# In[85]:


df1.nunique()


# In[87]:


df1.shape


# ##### For df2

# In[29]:


# Basic informationabout the dataset:
df2.info()


# In[31]:


# Summary 
df2.describe()


# In[33]:


df2.nunique()


# In[35]:


df2.shape


# ### 4. Data Cleaning  

# ##### (i) For df1 (Internal Data)

# ##### 1. Missing Values

# In[89]:


# Check the null values
df1.isnull().sum()


# In[91]:


# Fill the nan with median
df1[['Age_Oldest_TL', 'Age_Newest_TL']] = df1[['Age_Oldest_TL', 'Age_Newest_TL']].fillna(
    df1[['Age_Oldest_TL', 'Age_Newest_TL']].median()
)


# In[93]:


# Cross check
df1.isnull().sum()


# ##### 2. Duplicates

# In[95]:


# Checking duplicates in the data
df1.duplicated().sum()


# ##### (i) For df2 (External CIBIL Data)

# ##### 1. Missing Values

# In[97]:


# Check the null values
df2.isnull().sum()


# In[99]:


# Percentage of the null values in each column:
null_percentages = df2.isnull().mean() * 100

print("Percentage of null values in each column:")

print(null_percentages)


# In[101]:


# Percentage of the null values in each column: (Seperate the nulls)
null_percentages = df2.isnull().mean()* 100
print(null_percentages[(null_percentages > 0.01) & (null_percentages <= 50)])
#print("Percentage of null values in each column:")

# print(null_percentages)


# In[103]:


# Impute the columns which contains less than 50% missing values

# Step 1: Keep columns where % of -99999 is > 0 and <= 50
columns_to_impute = null_percentages[(null_percentages > 0) & (null_percentages <= 50)].index

# Step 2: Impute NaNs in columns with median for numeric columns and mode for categorical columns
for col in columns_to_impute:
    if df2[col].dtype in [np.number, 'float64', 'int64']:
        median_val = df2[col].median()
        df2[col].fillna(median_val, inplace=True)
    else:
        mode_val = df2[col].mode()[0]
        df2[col].fillna(mode_val, inplace=True)

print(f"Imputed columns: {list(columns_to_impute)}")


# In[105]:


df2.shape


# ##### 2. Duplicates

# In[107]:


# Checking duplicates in the data
df2.duplicated().sum()


# ### 5. Merge the dataframes

# In[109]:


# Checking common column names
for i in list(df1.columns):
    if i in list(df2.columns):
        print (i)


# In[111]:


# Merge the two dataframes, inner join so that no nulls are present
df = pd. merge ( df1, df2, how ='inner', left_on = ['PROSPECTID'], right_on = ['PROSPECTID'] )


# In[113]:


df.head(3)


# #### 1. Missing Values

# In[116]:


# Check the null values
df.columns[(df == -99999).any()]


# In[118]:


df.isnull().sum()


# #### 2. Duplicates

# In[121]:


# Check the duplicates
df.duplicated().sum()


# #### 3. Outliers

# In[124]:


# Check the outliers
for col in df.select_dtypes(include='number').columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()


# In[126]:


# Treat the Outliers
def outliers(x):
    if ((x.dtype == int) or (x.dtype == float)): 
        x = x.clip(lower = x.quantile(0.01), upper = x.quantile(0.99))
    return x


# In[128]:


df  = df.apply(lambda x: outliers(x))


# In[130]:


# Check the outliers
for col in df.select_dtypes(include='number').columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()


# #### # Still have the outliers in the data. That's why using the IQR Method to remove them

# In[134]:


def outliers_iqr(df):
    for col in df.select_dtypes(include='number').columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    return df

df = outliers_iqr(df)


# In[136]:


# Check the outliers
for col in df.select_dtypes(include='number').columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()


# In[138]:


# Check how many columns are categorical
for i in df.columns:
    if df[i].dtype == 'object':
        print(i)


# In[140]:


# Label encoding for the categorical features
['MARITALSTATUS', 'EDUCATION', 'GENDER' , 'last_prod_enq2' ,'first_prod_enq2']



df['MARITALSTATUS'].unique()    
df['EDUCATION'].unique()
df['GENDER'].unique()
df['last_prod_enq2'].unique()
df['first_prod_enq2'].unique()


# In[142]:


# Ordinal feature -- EDUCATION
# SSC            : 1
# 12TH           : 2
# GRADUATE       : 3
# UNDER GRADUATE : 3
# POST-GRADUATE  : 4
# OTHERS         : 1
# PROFESSIONAL   : 3

# Others has to be verified by the business end user 


df.loc[df['EDUCATION'] == 'SSC',['EDUCATION']]              = 1
df.loc[df['EDUCATION'] == '12TH',['EDUCATION']]             = 2
df.loc[df['EDUCATION'] == 'GRADUATE',['EDUCATION']]         = 3
df.loc[df['EDUCATION'] == 'UNDER GRADUATE',['EDUCATION']]   = 3
df.loc[df['EDUCATION'] == 'POST-GRADUATE',['EDUCATION']]    = 4
df.loc[df['EDUCATION'] == 'OTHERS',['EDUCATION']]           = 1
df.loc[df['EDUCATION'] == 'PROFESSIONAL',['EDUCATION']]     = 3


# In[144]:


df['EDUCATION'] = df['EDUCATION'].astype(int)


# In[146]:


df.shape


# In[148]:


df_encoded = pd.get_dummies(df, columns=['MARITALSTATUS','GENDER', 'last_prod_enq2' ,'first_prod_enq2'])

df_encoded.info()
k = df_encoded.describe()


# In[150]:


df_encoded.shape


# In[157]:


df_encoded.Approved_Flag.value_counts()


# In[159]:


df_encoded.Approved_Flag.value_counts()/df_encoded.Approved_Flag.value_counts().sum()


# ##### # It's the case of  Imbalance Data 

# #### Feature Engineering

# In[153]:


# Separate x & y variable
y = df_encoded['Approved_Flag']
X = df_encoded.drop(['Approved_Flag'], axis=1)


# In[164]:


X.corr() 


# In[166]:


X.std()/X.mean()


# In[168]:


# RFE - Recursive Feature Elimination 
classifier = RandomForestClassifier()
rfe  = RFE(estimator = classifier, n_features_to_select = 10)
rfe = rfe.fit(X,y)


# In[170]:


X.columns[rfe.support_]


# #### Select KBEST

# In[175]:


SKB = SelectKBest(f_classif, k=15)


# In[177]:


X_XKB = SKB.fit_transform(X, y)


# In[179]:


X.columns[SKB.get_support()]


# In[181]:


X_new = X[list(set(list(X.columns[rfe.support_]) + list(X.columns[SKB.get_support()])))]


# In[183]:


X_new.shape


# In[185]:


# Feature Scaling
scaler=StandardScaler()

X=scaler.fit_transform(X_new)


# In[195]:


X = pd.DataFrame(X, columns = ['Age_Oldest_TL', 'enq_L3m', 'Total_TL', 'num_std_6mts', 'num_std_12mts',
       'max_recent_level_of_deliq', 'Credit_Score', 'Tot_Closed_TL', 'num_std',
       'enq_L12m', 'tot_enq', 'PL_enq', 'PROSPECTID', 'PL_enq_L12m', 'enq_L6m',
       'Secured_TL', 'time_since_recent_enq'])


# ### 6. Machine Learning 
# #### Model Building

# #### 1. Logistic Regression

# In[201]:


# Separate x & y variable
y = df_encoded['Approved_Flag']
x = X  


# In[202]:


# Split the data into: Train & Test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[199]:


from sklearn.linear_model import LogisticRegression


# In[205]:


LR = LogisticRegression()
LR.fit(x_train, y_train)
pred = LR.predict(x_test)


# In[207]:


accuracy_score(y_test,pred)


# In[209]:


print(classification_report(y_test,pred))


# #### 2. Random Forest 

# In[212]:


# Separate x & y variable
y = df_encoded['Approved_Flag']
x = X  


# In[214]:


# Split the data into: Train & Test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[250]:


# Define & Fit the model
rf_classifier = RandomForestClassifier(n_estimators = 200, random_state=42)
rf_classifier.fit(x_train, y_train)
y_pred_rf = rf_classifier.predict(x_test)


# In[251]:


accuracy_score(y_test,y_pred_rf)


# In[254]:


print(classification_report(y_test,y_pred_rf))


# #### 3. XGBoost 

# In[258]:


import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

xgb_classifier = xgb.XGBClassifier(objective='multi:softmax',  num_class=4)


# In[259]:


# Separate the x & y variables
y = df_encoded['Approved_Flag']
x = X 


# In[262]:


# Lable encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


# In[264]:


# Split the data into: Train & test
x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)


# In[266]:


# Define & Fit the Model
xgb_classifier.fit(x_train, y_train)
y_pred_xgb = xgb_classifier.predict(x_test)


# In[268]:


accuracy_score(y_test,y_pred_xgb)  


# In[270]:


print(classification_report(y_test,y_pred_xgb))


# #### 4. Decision Tree

# In[272]:


from sklearn.tree import DecisionTreeClassifier


# In[274]:


# Separate the x & y variables
y = df_encoded['Approved_Flag']
x = X 


# In[276]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[278]:


dt_model = DecisionTreeClassifier(max_depth=20, min_samples_split=10)
dt_model.fit(x_train, y_train)
y_pred_dt = dt_model.predict(x_test)


# In[280]:


accuracy_score(y_test,y_pred_dt)


# In[282]:


print(classification_report(y_test,y_pred_dt))


# In[ ]:


# Import pickle and save the object
import pickle

# Save to a pickle file
with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

