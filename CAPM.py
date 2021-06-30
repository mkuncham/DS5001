#!/usr/bin/env python
# coding: utf-8

# ## Capital Asset Pricing Model (CAPM)
# ### Strength Training with Functions, Numpy
# 
# 
# ### University of Virginia
# ### DS 5100: Programming for Data Science
# ### Last Updated: March 18, 2021
# ---

# ### Objectives: 
# - Use numpy and functions to compute a stock's CAPM beta
# - Perform sensitivity analysis to understand how the data points impact the beta estimate
# 
# ### Background
# 
# 
# In finance, CAPM is a single-factor regression model used for explaining and predicting excess stock returns. There are better, more accurate models, but it has its uses. For example, the *market beta* is a useful output.
# 
# 
# Here is the formula for calculating the expected excess return:
# 
# \begin{aligned} &E[R_i] - R_f  = \beta_i ( E[R_m] - R_f ) \\ \\ &\textbf{where:} \\ &ER_i = \text{expected return of stock i} \\ &R_f = \text{risk-free rate} \\ &\beta_i = \text{beta of the stock} \\ &ER_m - R_f = \text{market risk premium} \\ \end{aligned} 
# 
# #### Review the instructions below to complete the requested tasks.
# 
# #### TOTAL POINTS: 10
# ---  
# 

# In[1]:


# load modules
import numpy as np
import pandas as pd

# risk-free Treasury rate
R_f = 0.0175 / 252


# In[4]:


# read in the market data
data = pd.read_csv('capm_market_data.csv')


# Look at some records  
# SPY is an ETF for the S&P 500 (the "stock market")  
# AAPL is Apple  
# The values are closing prices, adjusted for splits and dividends

# In[5]:


data.head()


# Drop the date column

# In[13]:


data = data.drop(['date'], axis=1)

data.head()


# Compute daily returns (percentage changes in price) for SPY, AAPL  
# Be sure to drop the first row of NaN  
# Hint: pandas has functions to easily do this

# In[16]:


daily_returns_all = data.pct_change()
daily_returns = daily_returns_all[1::]


# #### (1 PT) Print the first 5 rows of returns

# In[17]:


daily_returns.head()


# Save AAPL, SPY returns into separate numpy arrays  
# #### (1 PT) Print the first five values from the SPY numpy array, and the AAPL numpy array

# In[24]:


spy_returns= daily_returns['spy_adj_close'].to_numpy()
appl_returns= daily_returns['aapl_adj_close'].to_numpy()
print('spy', spy_returns[0:5], 'aapl', appl_returns[0:5])


# ##### Compute the excess returns of AAPL, SPY by simply subtracting the constant *R_f* from the returns.
# ##### Specifically, for the numpy array containing AAPL returns, subtract *R_f* from each of the returns. Repeat for SPY returns.
# 
# NOTE:  
# AAPL - *R_f* = excess return of Apple stock  
# SPY - *R_f* = excess return of stock market
# 

# In[26]:


appl_excess = appl_returns - R_f
appl_excess[0:5]


# In[27]:


spy_excess = spy_returns - R_f
spy_excess[0:5]


# #### (1 PT) Print the LAST five excess returns from both AAPL, SPY numpy arrays
# 

# In[29]:


#APPL
print(appl_excess[-5:])


# In[30]:


#SPY
print(spy_excess[-5:])


# #### (1 PT) Make a scatterplot with SPY excess returns on x-axis, AAPL excess returns on y-axis####
# Matplotlib documentation: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html

# In[37]:


import matplotlib.pyplot as plt
plt.scatter(spy_excess, appl_excess)


# #### (3 PTS) Use Linear Algebra (matrices) to Compute the Regression Coefficient Estimate, \\(\hat\beta_i\\)
# 
# Hint 1: Here is the matrix formula where *x′* denotes transpose of *x*.
# 
# \begin{aligned} \hat\beta_i=(x′x)^{−1}x′y \end{aligned} 
# 
# Hint 2: consider numpy functions for matrix multiplication, transpose, and inverse. Be sure to review what these operations do, and how they work, if you're a bit rusty.

# In[101]:


x = spy_excess
y = appl_excess
beta = ((np.matmul(x.transpose(),x))**-1)*(np.matmul(x.transpose(),y))
beta


# You should have found that the beta estimate is greater than one.  
# This means that the risk of AAPL stock, given the data, and according to this particular (flawed) model,  
# is higher relative to the risk of the S&P 500.
# 

# 

# #### Measuring Beta Sensitivity to Dropping Observations (Jackknifing)

# Let's understand how sensitive the beta is to each data point.   
# We want to drop each data point (one at a time), compute \\(\hat\beta_i\\) using our formula from above, and save each measurement.
# 
# #### (3 PTS) Write a function called `beta_sensitivity()` with these specs:
# 
# - take numpy arrays x and y as inputs
# - output a list of tuples. each tuple contains (observation row dropped, beta estimate)
# 
# Hint: **np.delete(x, i).reshape(-1,1)** will delete observation i from array x, and make it a column vector

# In[117]:


def beta_sensitivity(x,y):
    x_safe = x
    y_safe = y
    tups = []
    tot = len(x)
    beta = 0
    for r in range(tot):
        x = np.delete(x, r).reshape(-1,1)
        y = np.delete(y, r).reshape(-1,1)
        beta = np.dot(((np.matmul(x.T,x))**-1),(np.matmul(x.T,y)))
        tups.append((r,beta))
        print(len(x),len(y))
#         np.delete(x, r).reshape(-1,1)
#         np.delete(y, r).reshape(-1,1)
        #print(len(x),len(y))
        
    return tups
        
        
    


# #### Call `beta_sensitivity()` and print the first five tuples of output.

# In[118]:


betas = beta_sensitivity(x, y)
betas


# The beta does not seem sensitive? 

# In[ ]:




