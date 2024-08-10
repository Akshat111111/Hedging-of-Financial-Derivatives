#!/usr/bin/env python
# coding: utf-8

# # Dow Jones Industrial Average (DJIA) prediction 
# We will be predicting the DJIA closing value by using the top 25 headlines for the day.
# 
# The Dow Jones Industrial Average, Dow Jones, or simply the Dow, is a stock market index that measures the stock performance of 30 large companies listed on stock exchanges in the United States.
# 
# ![image.png](attachment:image.png)
# 
# 
# 

# # Step 1 - Importing all the libraries 

# In[6]:


import time
start = time.time()
import os

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import cross_val_score #sklearn features various ML algorithms
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from nltk.sentiment.vader import SentimentIntensityAnalyzer #sentiment analysis 
from textblob import TextBlob #processing textual data

#plotting
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

#statistics & econometrics
import statsmodels.tsa.api as smt
import statsmodels.api as sm

#model fiiting and selection
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor


# # Step 2 - Importing the dataset

# In[7]:


df = pd.read_csv("Combined_News_DJIA.csv", sep=',',low_memory=False,
                    parse_dates=[0])

full_stock = pd.read_csv("upload_DJIA_table.csv",low_memory=False,
                    parse_dates=[0])

#add the closing stock value to the df - this will be the y variable
df["Close"]=full_stock.Close

#show how the dataset looks like
df.head(5)


# In[8]:


#drop the label column
df2 = df.drop(['Label'], axis=1)

#view the new dataframe
df2.head(10)


# # Step 3 - Data cleaning
# NA treatment :
# We'll simply fill the NAs in the numerical features (Date, Close). In the text features we'll fill the missing values with ''.

# In[9]:


#check for NAN
df2.isnull().sum()


# There are a few headlines missing. Let's fill them in with a whitespace.

# In[10]:


#replacing nan values with a whitespace
df2 = df2.replace(np.nan, ' ', regex=True)

#sanity check
df2.isnull().sum().sum()


# Remove the HTML tags :
# There are several non-word tags in the headlines that would just bias the sentiment analysis so we need to remove them and replace with ''. This can be done with regex.

# In[11]:


#Remove the HTML tags
df2 = df2.replace('b\"|b\'|\\\\|\\\"', '', regex=True)
df2.head(2)


# Sentiment and subjectivity score extraction :
# Now we run the sentiment analysis extracting the compound score that goes from -0.5 (most negative) to 0.5 (most positive). I'm going to use the "dirty" texts in this part because VADER can utilize the information such as ALL CAPS, punctuation, etc. We'll also calculate the subjectivity of each headline using the TextBlob package.
# 
# Initialise the VADER analyzer.

# In[12]:


#Sentiment and subjectivity score extraction
Anakin = SentimentIntensityAnalyzer()

Anakin.polarity_scores(" ")


# Write a function to save the subjectivity score directly from TextBlob function's output. Subjectivity score might detect direct quotes in the headlines and positive stuff is rarely quoted in the headline.

# In[13]:


def detect_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity

detect_subjectivity(" ") #should return 0


# In[14]:


start_vect=time.time()
print("ANAKIN: 'Intializing the process..'")

#get the name of the headline columns
cols = []
for i in range(1,26):
    col = ("Top{}".format(i))
    cols.append(col)


for col in cols:
    df2[col] = df2[col].astype(str) # Make sure data is treated as a string
    df2[col+'_comp']= df2[col].apply(lambda x:Anakin.polarity_scores(x)['compound'])
    df2[col+'_sub'] = df2[col].apply(detect_subjectivity)
    print("{} Done".format(col))
    
print("VADER: Vaderization completed after %0.2f Minutes"%((time.time() - start_vect)/60))


# In[15]:


#the text isn't required anymore
df2 = df2.drop(cols,axis=1)
df2.head(5)


# Summarise the compound and subjectivity scores weighted by rating of the headline (top1 has the most weight)

# In[16]:


comp_cols = []
for col in cols:
    comp_col = col + "_comp"
    comp_cols.append(comp_col)

w = np.arange(1,26,1).tolist()
w.reverse()

weighted_comp = []
max_comp = []
min_comp = []
for i in range(0,len(df)):
    a = df2.loc[i,comp_cols].tolist()
    weighted_comp.append(np.average(a, weights=w))
    max_comp.append(max(a))
    min_comp.append(min(a))

df2['compound_mean'] = weighted_comp
df2['compound_max'] = max_comp
df2['compound_min'] = min_comp


sub_cols = []
for col in cols:
    sub_col = col + "_sub"
    sub_cols.append(sub_col)
    
weighted_sub = []
max_sub = []
min_sub = []
for i in range(0,len(df2)):
    a = df2.loc[i,sub_cols].tolist()
    weighted_sub.append(np.average(a, weights=w))
    max_sub.append(max(a))
    min_sub.append(min(a))

df2['subjectivity_mean'] = weighted_sub
df2['subjectivity_max'] = max_sub
df2['subjectivity_min'] = min_sub

to_drop = sub_cols+comp_cols
df2 = df2.drop(to_drop, axis=1)

df2.head(5)


# # Step 4. Exploratory Data Analysis
# First, the timeseries of the y (to be predicted) variable will be explored. It's likely that the timeseries isn't stationary which however doesn't worry us in this case as the models won't be of the classical timeseries methods family.

# In[17]:


#EDA-time series plot
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df2.Date, y=df.Close,
                    mode='lines'))
title = []
title.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                              xanchor='left', yanchor='bottom',
                              text='Development of stock values from Aug, 2008 to Jun, 2016',
                              font=dict(family='Arial',
                                        size=30,
                                        color='rgb(37,37,37)'),
                              showarrow=False))
fig1.update_layout(xaxis_title='Date',
                   yaxis_title='Closing stock value (in $)',
                  annotations=title)
fig1.show()


# It is quite obvious that the timeseries isn't stationary at all. It just seems to be a downwards trend over the time.
# So let's look at how much unstationary the timeseries actually is using Dickey-Fuller Test.
# 
# In Dickey-Fuller test, we test the null hypothesis that a unit root is present in time series data. 
# To make things a bit more clear, this test is checking for stationarity or non-stationary data.  The test is trying to reject the null hypothesis that a unit root exists and the data is non-stationary. 

# In[18]:


#function for quick plotting and testing of stationarity
def stationary_plot(y, lags=None, figsize=(12, 7), style='bmh'):
    """
        Plot time series, its ACF and PACF, calculate Dickey–Fuller test
        
        y - timeseries
        lags - how many lags to include in ACF, PACF calculation
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
        
    with plt.style.context(style):    
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        
        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        plt.tight_layout()


# In[19]:


stationary_plot(df2.Close)


# Next we look at the compound sentiment scores.

# In[20]:


fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df2.Date, y=df2.compound_mean,
                    mode='lines',
                    name='Mean'))
fig2.add_trace(go.Scatter(x=df2.Date, y=df2.compound_max,
                    mode='lines',
                    name='Maximum'))
fig2.add_trace(go.Scatter(x=df2.Date, y=df2.compound_min,
                    mode='lines',
                    name='Minimum'))
title = []
title.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                              xanchor='left', yanchor='bottom',
                              text='Development of sentiment compound score',
                               font=dict(family='Arial',
                                       size=30,
                                        color='rgb(37,37,37)'),
                              showarrow=False))
fig2.update_layout(xaxis_title='Date',
                   yaxis_title='Compound score',
                  annotations=title)
fig2.show()


# Let's also plot the distribution of the compound score.

# In[21]:


compm_hist = px.histogram(df2, x="compound_mean")
compm_hist.show()


# Now, we take a look at the subjectivity scores.

# In[22]:


fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=df2.Date, y=df2.subjectivity_mean,
                    mode='lines',
                    name='Mean'))
fig3.add_trace(go.Scatter(x=df2.Date, y=df2.subjectivity_min,
                    mode='lines',
                    name='Min'))
fig3.add_trace(go.Scatter(x=df2.Date, y=df2.subjectivity_max,
                    mode='lines',
                    name='Max'))
title = []
title.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                              xanchor='left', yanchor='bottom',
                              text='Development of subjectivity score',
                              font=dict(family='Arial',
                                        size=30,
                                        color='rgb(37,37,37)'),
                              showarrow=False))
fig3.update_layout(xaxis_title='Date',
                   yaxis_title='Subjectivity score',
                  annotations=title)
fig3.show()


# Now we plot distribution of the subjectivity scores as well.

# In[23]:


subm_hist = px.histogram(df2, x="subjectivity_mean")
subm_hist.show()


# Next we'll look at some descriptive statistics about the data.

# In[24]:


df2.describe()


# # Feature selection
# We are not going to use many FS methods since the features were mostly handcrafted. So we'll simply look at their variance and proportion of unique values.

# In[25]:


def unique_ratio (col):
    return len(np.unique(col))/len(col)

cols = ['Close', 'compound_mean', 'compound_max', 'compound_min', 'subjectivity_mean', 'subjectivity_max', 'subjectivity_min']

ur = []
var = []
for col in cols:
    ur.append(unique_ratio(df2[col]))
    var.append(np.var(df2[col]))
    
feature_sel = pd.DataFrame({'Column': cols, 
              'Unique': ur,
              'Variance': var})
feature_sel


# In[26]:


sel_fig = go.Figure(data=go.Scatter(
    x=feature_sel.Column,
    y=feature_sel.Unique,
    mode='markers',
    marker=dict(size=(feature_sel.Unique*100)),
))
sel_fig.update_layout(title='Ratio of unique values', 
                      yaxis_title='Unique ratio')
sel_fig.show()


# Compound maximum and minimum are potentially less interesting as only ~18% of their values are unique. Also maximum of subjectivity has very low values. Minimum subjectivity has contains almost only 0. We'll drop the subjectivity min and max.

# In[27]:


drop = ['subjectivity_min', 'subjectivity_max']
clean_df = df2.drop(drop,axis=1)


# # Step 5 - Lag the extracted features
# To allow the models to look into the past, we'll add features which are essentially just copies of rows from n-steps back. In order to not create too many new features we'll add only features from 1 week prior to the current datapoint.

# In[28]:


lag_df = clean_df.copy()
lag_df.head(3)


# In[29]:


to_lag = list(lag_df.columns)
to_lag_4 = to_lag[1]
to_lag_1 = to_lag[2:len(to_lag)]


# In[30]:


#lagging text features two days back
for col in to_lag_1:
    for i in range(1,3):
        new_name = col + ('_lag_{}'.format(i))
        lag_df[new_name] = lag_df[col].shift(i)
    
#lagging closing values 4 days back
for i in range(1, 5):
    new_name = to_lag_4 + ('_lag_{}'.format(i))
    lag_df[new_name] = lag_df[to_lag_4].shift(i)


# In this process, rows with NAs were created. Unfortunately these rows will have to be removed since we simply don't have the data from the future.

# In[31]:


#Show many rows need to be removed
lag_df.head(10) 


# Above we can see that the first 4 rows now have missing values. Let's delete them and reset index.

# In[32]:


lag_df = lag_df.drop(lag_df.index[[np.arange(0,4)]])
lag_df = lag_df.reset_index(drop=True)

#sanity check for NaNs
lag_df.isnull().sum().sum()


# In[33]:


lag_df.head(5)


# # Step 8 - Model training
# Let's train 3 ML models. We'll do this in 2 rounds. First, using the econometric features alone (7 lags of y). 
# Second, including the information extracted from the headlines (compound, subjectivity and their lags)
# 
# Models:
# 
# 1)Ridge regression - punish model for using too many features but doesn't allow the coeficients drop to zero completely
# 
# 2)Random forest
# 
# 3)XGBoost
# 
# We'll score all models by mean squared error as it gives higher penalty to larger mistakes. And before each model training we'll standardize the training data.
# 
# The first step will be creating folds for cross-validation. We'll use the same folds for all models in order to allow for creating a meta-model. Since we're working with timeseries the folds cannot be randomly selected. Instead a fold will be a sequence of data so that we don't lose the time information.

# In[34]:


# for time-series cross-validation set 10 folds 
tscv = TimeSeriesSplit(n_splits=10)


# The cost function to minimize is mean squared error because this function assigns cost proportionally to the error size. The mean absolute percentage error will be used for plotting and easier interpretation. It's much easier to understand the errors of a model in terms of percentage. Each training set is scaled (normalized) independently to minimize data leakage.

# In[35]:


def mape(y_true, y_pred): 
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
scorer = make_scorer(mean_squared_error)
scaler = StandardScaler()   


# Next we split the dataset into training and testing. 20% of the data will be used for testing.

# In[36]:


def ts_train_test_split(X, y, test_size):
    """
        Perform train-test split with respect to time series structure
    """
    
    # get the index after which test set starts
    test_index = int(len(X)*(1-test_size))
    
    X_train = X.iloc[:test_index]
    y_train = y.iloc[:test_index]
    X_test = X.iloc[test_index:]
    y_test = y.iloc[test_index:]
    
    return X_train, X_test, y_train, y_test


# In[37]:


X = lag_df.drop(['Close'],axis=1)
X.index = X["Date"]
X = X.drop(['Date'],axis=1)
y = lag_df.Close

X_train, X_test, y_train, y_test = ts_train_test_split(X, y, test_size = 0.2)

#sanity check
(len(X_train)+len(X_test))==len(X)


# In[38]:


#function for plotting coeficients of models (lasso and XGBoost)
def plotCoef(model,train_x):
    """
        Plots sorted coefficient values of the model
    """
    
    coefs = pd.DataFrame(model.coef_, train_x.columns)
    coefs.columns = ["coef"]
    coefs["abs"] = coefs.coef.apply(np.abs)
    coefs = coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)
    
    plt.figure(figsize=(15, 7))
    coefs.coef.plot(kind='bar')
    plt.grid(True, axis='y')
    plt.hlines(y=0, xmin=0, xmax=len(coefs), linestyles='dashed');


# # Step 8.1 - Econometric models
# First let's train models using only the lags of the y variable (i.e. Close).

# In[39]:


econ_cols = list(X_train.columns)
econ_cols = econ_cols[12:17]
X_train_e = X_train[econ_cols]
X_test_e = X_test[econ_cols]
y_train_e = y_train
y_test_e = y_test


# In[40]:


econ_perf = pd.DataFrame(columns=['Model','MSE', 'SD'])
econ_perf


# # Ridge regression

# In[41]:


ridge_param = {'model__alpha': list(np.arange(0.001,1,0.001))}
ridge = Ridge(max_iter=5000)
pipe = Pipeline([
    ('scale', scaler),
    ('model', ridge)])
search_ridge = GridSearchCV(estimator=pipe,
                          param_grid = ridge_param,
                          scoring = scorer,
                          cv = tscv,
                          n_jobs=4
                         )
search_ridge.fit(X_train_e, y_train_e)


# In[42]:


ridge_e = search_ridge.best_estimator_

#get cv results of the best model + confidence intervals
from sklearn.model_selection import cross_val_score
cv_score = cross_val_score(ridge_e, X_train_e, y_train_e, cv=tscv, scoring=scorer)
econ_perf = econ_perf.append({'Model':'Ridge', 'MSE':np.mean(cv_score), 'SD':(np.std(cv_score))}, ignore_index=True)
ridge_e


# In[43]:


plotCoef(ridge_e['model'], X_train_e)


# In[44]:


coefs = ridge_e['model'].coef_
ridge_coefs = pd.DataFrame({'Coef': coefs,
                           'Name': list(X_train_e.columns)})
ridge_coefs["abs"] = ridge_coefs.Coef.apply(np.abs)
ridge_coefs = ridge_coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)
ridge_coefs


# In[45]:


econ_perf


# # Random forest

# In[46]:


rf_param = {'model__n_estimators': [10, 100, 300],
            'model__max_depth': [10, 20, 30, 40],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 3],
            'model__max_features': ["auto", 'sqrt']}


# In[47]:


rf = RandomForestRegressor()
pipe = Pipeline([
    ('scale', scaler),
    ('model', rf)])
gridsearch_rf = GridSearchCV(estimator=pipe,
                          param_grid = rf_param,
                          scoring = scorer,
                          cv = tscv,
                          n_jobs=4,
                          verbose=3
                         )


# In[48]:


gridsearch_rf.fit(X_train_e, y_train_e)


# In[49]:


rf_e = gridsearch_rf.best_estimator_

#get cv results of the best model + confidence intervals
cv_score = cross_val_score(rf_e, X_train_e, y_train_e, cv=tscv, scoring=scorer)
econ_perf = econ_perf.append({'Model':'RF', 'MSE':np.mean(cv_score), 'SD':(np.std(cv_score))}, ignore_index=True)


# In[50]:


econ_perf


# # XGBoost
# Using linear booster because tree methods don't work with timeseries very well

# In[52]:


xgb_param = {'model__lambda': list(np.arange(0.1,3, 0.1)), #L2 regularisation
             'model__alpha': list(np.arange(0.1,3, 0.1)),  #L1 regularisation
            }


# In[53]:


xgb = XGBRegressor(booster='gblinear', feature_selector='shuffle', objective='reg:squarederror')

pipe = Pipeline([
    ('scale', scaler),
    ('model', xgb)])
gridsearch_xgb = GridSearchCV(estimator=pipe,
                          param_grid = xgb_param,
                          scoring = scorer,
                          cv = tscv,
                          n_jobs=4,
                          verbose=3
                         )


# In[54]:


gridsearch_xgb.fit(X_train_e, y_train_e)


# In[55]:


xgb_e = gridsearch_xgb.best_estimator_

#get cv results of the best model + confidence intervals
cv_score = cross_val_score(xgb_e, X_train_e, y_train_e, cv=tscv, scoring=scorer)
econ_perf = econ_perf.append({'Model':'XGB', 'MSE':np.mean(cv_score), 'SD':(np.std(cv_score))}, ignore_index=True)
xgb_e


# In[56]:


print(econ_perf)


# In[57]:


econ_fig = px.scatter(econ_perf, x="Model", y='MSE', color='Model', error_y="SD")
econ_fig.update_layout(title_text="Performance of models trained on lags of y")
econ_fig.show()


# # Step 8.2 - NLP models
# Let's try now predict the stock value using only information from the news headlines.

# In[58]:


X_train_n = X_train.drop(econ_cols, axis=1)
X_test_n = X_test.drop(econ_cols, axis=1)
y_train_n = y_train
y_test_n = y_test


# In[59]:


nlp_perf = pd.DataFrame(columns=['Model','MSE', 'SD'])
nlp_perf


# 
# # Ridge regression

# In[60]:


ridge_param = {'model__alpha': list(np.arange(1,10,0.1))}
ridge = Ridge(max_iter=5000)
pipe = Pipeline([
    ('scale', scaler),
    ('model', ridge)
])
search_ridge = GridSearchCV(estimator=pipe,
                          param_grid = ridge_param,
                          scoring = scorer,
                          cv = tscv,
                          n_jobs=4
                         )
search_ridge.fit(X_train_n, y_train_n)


# In[61]:


ridge_n = search_ridge.best_estimator_

#get cv results of the best model + confidence intervals
cv_score = cross_val_score(ridge_n, X_train_n, y_train_n, cv=tscv, scoring=scorer)
nlp_perf = nlp_perf.append({'Model':'Ridge', 'MSE':np.mean(cv_score), 'SD':(np.std(cv_score))}, ignore_index=True)
ridge_n


# In[62]:


plotCoef(ridge_n['model'], X_train_n)

coefs = ridge_n['model'].coef_
ridge_coefs = pd.DataFrame({'Coef': coefs,
                           'Name': list(X_train_n.columns)})
ridge_coefs["abs"] = ridge_coefs.Coef.apply(np.abs)
ridge_coefs = ridge_coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)
ridge_coefs


# In[63]:


mape(y_test, ridge_n.predict(X_test_n))


# Predicted values for years 2014-2016 using ridge regression.

# In[64]:


print(ridge_n.predict(X_test_n))


# # Random Forest

# In[65]:


rf_param = {'model__n_estimators': [10, 100, 300],
            'model__max_depth': [10, 20, 30, 40],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 3],
            'model__max_features': ["auto", 'sqrt']}
rf = RandomForestRegressor()
pipe = Pipeline([
    ('scale', scaler),
    ('model', rf)])
gridsearch_rf = GridSearchCV(estimator=pipe,
                          param_grid = rf_param,
                          scoring = scorer,
                          cv = tscv,
                          n_jobs=4,
                          verbose=3
                         )
gridsearch_rf.fit(X_train_n, y_train_n)


# In[66]:


rf_n = gridsearch_rf.best_estimator_

#get cv results of the best model + confidence intervals
cv_score = cross_val_score(rf_n, X_train_n, y_train_n, cv=tscv, scoring=scorer)
nlp_perf = nlp_perf.append({'Model':'RF', 'MSE':np.mean(cv_score), 'SD':(np.std(cv_score))}, ignore_index=True)


# # XGBoost

# In[67]:


xgb_param = {'model__lambda': list(np.arange(1,10, 1)), #L2 regularisation
             'model__alpha': list(np.arange(1,10, 1)),  #L1 regularisation
            }
xgb = XGBRegressor(booster='gblinear', feature_selector='shuffle', objective='reg:squarederror')

pipe = Pipeline([
    ('scale', scaler),
    ('model', xgb)])
gridsearch_xgb = GridSearchCV(estimator=pipe,
                          param_grid = xgb_param,
                          scoring = scorer,
                          cv = tscv,
                          n_jobs=4,
                          verbose=3
                         )
gridsearch_xgb.fit(X_train_n, y_train_n)


# In[68]:


xgb_n = gridsearch_xgb.best_estimator_

#get cv results of the best model + confidence intervals
cv_score = cross_val_score(xgb_n, X_train_n, y_train_n, cv=tscv, scoring=scorer)
nlp_perf = nlp_perf.append({'Model':'XGB', 'MSE':np.mean(cv_score), 'SD':(np.std(cv_score))}, ignore_index=True)
xgb_n


# In[69]:


print(nlp_perf)


# In[70]:


nlp_fig = px.scatter(nlp_perf, x="Model", y='MSE', color='Model', error_y="SD")
#nlp_fig.update_layout(title_text="Performance of models trained on NLP features",
nlp_fig.show()


# The 3 models are performing quite similary. They might be useful candidates for stacking.

# # Econ+NLP models
# Let's use all features now

# # Ridge Regression

# In[71]:


en_perf = pd.DataFrame(columns=['Model','MSE', 'SD'])
en_perf


# In[72]:


ridge_param = {'model__alpha': list(np.arange(0.1,1,0.01))}
ridge = Ridge(max_iter=5000)
pipe = Pipeline([
    ('scale', scaler),
    ('model', ridge)])
search_ridge = GridSearchCV(estimator=pipe,
                          param_grid = ridge_param,
                          scoring = scorer,
                          cv = tscv,
                          n_jobs=4,
                          verbose=3
                         )
search_ridge.fit(X_train, y_train)


# In[73]:


ridge_en = search_ridge.best_estimator_

#get cv results of the best model + confidence intervals
cv_score = cross_val_score(ridge_en, X_train, y_train, cv=tscv, scoring=scorer)
en_perf = en_perf.append({'Model':'Ridge', 'MSE':np.mean(cv_score), 'SD':(np.std(cv_score))}, ignore_index=True)
ridge_en


# In[74]:


coefs = ridge_en['model'].coef_
ridge_coefs = pd.DataFrame({'Coef': coefs,
                           'Name': list(X_train.columns)})
ridge_coefs["abs"] = ridge_coefs.Coef.apply(np.abs)
ridge_coefs = ridge_coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)
ridge_coefs


# In[75]:


plotCoef(ridge_en['model'], X_train)


# # Random Forest

# In[76]:


rf_param = {'model__n_estimators': [10, 100, 300],
            'model__max_depth': [10, 20, 30, 40],
            'model__min_samples_split': [2, 5, 10],
            'model__min_samples_leaf': [1, 2, 3],
            'model__max_features': ["auto", 'sqrt']}
rf = RandomForestRegressor()
pipe = Pipeline([
    ('scale', scaler),
    ('model', rf)])
gridsearch_rf = GridSearchCV(estimator=pipe,
                          param_grid = rf_param,
                          scoring = scorer,
                          cv = tscv,
                          n_jobs=4,
                          verbose=3
                         )
gridsearch_rf.fit(X_train, y_train)


# In[77]:


rf_en = gridsearch_rf.best_estimator_

#get cv results of the best model + confidence intervals
cv_score = cross_val_score(rf_en, X_train, y_train, cv=tscv, scoring=scorer)
en_perf = en_perf.append({'Model':'RF', 'MSE':np.mean(cv_score), 'SD':(np.std(cv_score))}, ignore_index=True)
rf_en


# # XGBoost

# In[78]:


xgb_param = {'model__lambda': list(np.arange(1,10, 1)), #L2 regularisation
             'model__alpha': list(np.arange(1,10, 1)),  #L1 regularisation
            }
xgb = XGBRegressor(booster='gblinear', feature_selector='shuffle', objective='reg:squarederror')

pipe = Pipeline([
    ('scale', scaler),
    ('model', xgb)])
gridsearch_xgb = GridSearchCV(estimator=pipe,
                          param_grid = xgb_param,
                          scoring = scorer,
                          cv = tscv,
                          n_jobs=4,
                          verbose=3
                         )
gridsearch_xgb.fit(X_train, y_train)


# In[79]:


xgb_en = gridsearch_xgb.best_estimator_

#get cv results of the best model + confidence intervals
cv_score = cross_val_score(xgb_en, X_train, y_train, cv=tscv, scoring=scorer)
en_perf = en_perf.append({'Model':'XGB', 'MSE':np.mean(cv_score), 'SD':(np.std(cv_score))}, ignore_index=True)
xgb_en


# # Trying to stack econometric and NLP models

# In[80]:


from sklearn.model_selection import cross_val_predict

X_train_stack = pd.DataFrame(pd.DataFrame(columns=['econ_r', 'nlp_r']))
X_train_stack['econ_r'] = cross_val_predict(ridge_e, X_train_e, y_train, cv=10)
X_train_stack['nlp_r'] = cross_val_predict(ridge_n, X_train_n, y_train, cv=10)

X_test_stack = pd.DataFrame(pd.DataFrame(columns=['econ_r', 'nlp_r']))
X_test_stack['econ_r'] = ridge_e.predict(X_test_e)
X_test_stack['nlp_r'] = ridge_n.predict(X_test_n)

X_train_stack.to_csv("Stack_train.csv")
X_test_stack.to_csv("Stack_test.csv")

from sklearn.linear_model import ElasticNetCV
stack = ElasticNetCV(cv=tscv)
stack.fit(X_train_stack, y_train)
cv_score = cross_val_score(stack, X_train_stack, y_train, cv=tscv, scoring=scorer)
stack_performance = {'Model':'XGB', 'MSE':np.mean(cv_score), 'SD':(np.std(cv_score))}
stack_performance

mape(y_test, stack.predict(X_test_stack))


# In[81]:


coefs = stack.coef_
ridge_coefs = pd.DataFrame({'Coef': coefs,
                           'Name': list(X_train_stack.columns)})
ridge_coefs["abs"] = ridge_coefs.Coef.apply(np.abs)
ridge_coefs = ridge_coefs.sort_values(by="abs", ascending=False).drop(["abs"], axis=1)
print(ridge_coefs)
plotCoef(stack, X_train_stack)


# # Model Comparison

# prediction_compare = pd.DataFrame(pd.DataFrame(columns=['y_true', 'econ_r', 'econ_rf', 'econ_x', 'nlp_r', 'nlp_rf', 'nlp_x', 'comb_r', 'comb_rf', 'comb_x', 'stack']))
# prediction_compare['y_true'] = y_test
# prediction_compare['econ_r'] = ridge_e.predict(X_test_e)
# prediction_compare['econ_rf'] = rf_e.predict(X_test_e)
# prediction_compare['econ_x'] = xgb_e.predict(X_test_e)
# prediction_compare['nlp_r'] = ridge_n.predict(X_test_n)
# prediction_compare['nlp_rf'] = rf_n.predict(X_test_n)
# prediction_compare['nlp_x'] = xgb_n.predict(X_test_n)
# prediction_compare['comb_r'] = ridge_en.predict(X_test)
# prediction_compare['comb_rf'] = rf_en.predict(X_test)
# prediction_compare['comb_x'] = xgb_en.predict(X_test)
# prediction_compare['stack'] = stack.predict(X_test_stack)
# 
# prediction_compare.sample(3)

# # Outcomes
# ->The market is arguably to be a random walk. Although there is some direction to its movements, there is still quite a bit of randomness to its movements.
# 
# ->The news that we used might not be the most relevant. Perhaps it would have been better to use news relating to the 30 companies that make up the Dow.
# 
# ->More information could have been included in the model, such as the previous day(s)'s change, the previous day(s)'s main headline(s).
# 
# ->Many people have worked on this task for years and companies spend millions of dollars to try to predict the movements of the market, so we shouldn't expect anything too great considering the small amount of data that we are working with and the simplicity of our model.
