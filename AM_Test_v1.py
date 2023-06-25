#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


pd.set_option('display.max_columns', None)


# ### Split the P&L and Balance Sheet tabs in different workbooks

# #### Transposing the P&L sheet and then appending them together

# In[3]:


pl_dfs = pd.read_excel('Test-PL.xlsx', sheet_name = None, header = None)


# In[4]:


transposed_dfs_pl = []


# In[5]:


for sheet_name, df in pl_dfs.items():
    transposed_df_pl = df.set_index(0).transpose()
    transposed_dfs_pl.append(transposed_df_pl)


# In[6]:


pl_df = pd.concat(transposed_dfs_pl, ignore_index = True)


# In[7]:


pl_df


# #### Transposing the Balance sheet and then appending them together

# In[8]:


bs_dfs = pd.read_excel('Test-BS.xlsx', sheet_name = None, header = None)


# In[9]:


transposed_dfs_bs = []


# In[10]:


for sheet_name, df in bs_dfs.items():
    transposed_df_bs = df.set_index(0).transpose()
    transposed_dfs_bs.append(transposed_df_bs)


# In[11]:


bs_df = pd.concat(transposed_dfs_bs, ignore_index = True)


# In[12]:


bs_df


# In[13]:


# bs_df.to_csv("Bal.csv")
# pl_df.to_csv("PL.csv")


# In[14]:


df_comb = pd.merge(bs_df, pl_df, on = 'UEN', how = 'inner')


# In[15]:


label_df = pd.read_csv('test_label.csv')
label_df


# In[16]:


test = pd.merge(pd.merge(pl_df, bs_df, on = 'UEN'), label_df, on = 'UEN')


# In[17]:


test


# ### Data Wrangling on Test data

# In[18]:


test.dtypes


# In[19]:


col = ['Revenue', 'Sales', 'other Rev', 'Sub-contractor fees', 'Purchase & Direct Cost', 'total other Income', 'Employers CPF & SDL', 'Salaries & Bonus', "Director's Salary", 'Total operating expense', 'Total Amortization and Depreciation', 'Total Financial Charges', 'Total Selling and Distribution expense', 'Tax', 'Net Income', 'Cash', 'Receivables', 'Inventories', 'Total Current Asset', 'Total Fixed Asset', 'Intangible Asset', 'Trade Payables', 'Short Term Loan', 'Hire Purchase Creditors - Short term', 'Income Tax Payables', 'Amount owing to directors/shareholders/related parties', 'Total Current Liabilities', 'Long Term Loan', 'Hire Purchase Creditor_Long', 'Total Liabilities', 'Paid Up Capital', 'Retained Earning', 'Total Equity', 'Label']


# In[20]:


test[col] = test[col].apply(pd.to_numeric, errors='coerce')


# In[21]:


test.dtypes


# In[22]:


test.isnull().sum()


# In[23]:


test.rename(columns = {'Purchase & Direct Cost':'COGS', 'Hire Purchase Creditors - Short term':'Hire Purchase Creditors','Short Term Loan': 'Term Loan'}, inplace = True)


# In[24]:


cols_to_convert = ['total other Income', 'Total operating expense', 'Total Amortization and Depreciation', 'Total Financial Charges', 'Total Selling and Distribution expense', 'Tax', 'Cash', 'Intangible Asset','Inventories','Trade Payables', 'Term Loan', 'Hire Purchase Creditors', 'Income Tax Payables', 'Amount owing to directors/shareholders/related parties', 'Long Term Loan', 'Hire Purchase Creditor_Long', 'Paid Up Capital']
test[cols_to_convert] = test[cols_to_convert].replace(np.nan, 0)


# In[25]:


test.isnull().sum()


# In[26]:


test['Total salary and expense'] = test['Employers CPF & SDL'] + test['Salaries & Bonus'] + test["Director's Salary"] + test['Sub-contractor fees']

test['Total Asset'] = test['Total Current Asset'] + test['Total Fixed Asset'] + test['Intangible Asset']

test['Gross Profit'] = test['Sales'] + test['other Rev'] - test['COGS']


# In[27]:


test = test.drop(['Employers CPF & SDL','Salaries & Bonus',"Director's Salary",'Sub-contractor fees'], axis=1)


# In[28]:


# test.to_csv("test.csv")


# ### Adding EBIT and EBITDA

# In[29]:


test['EBITDA'] = test['Net Income'] + test['Tax'] + test['Total Amortization and Depreciation']

test['EBIT'] = test['Net Income'] + test['Tax']


# ### Adding Financial Ratios

# In[30]:


#Liquidity Ratios
test['Current Ratio'] = test['Total Current Asset']/test['Total Current Liabilities']

test['Cash Ratio'] = test['Cash']/test['Total Current Liabilities']

#Profitability Ratios
test['Gross Margin Ratio'] = test['Gross Profit']/test['Sales']

test['Operating Margin Ratio'] = test['EBIT']/test['Revenue']

test['Net Profit Ratio'] = test['Net Income']/test['Sales']

test['Net Profit to TA Ratio'] = test['Net Income']/test['Total Asset']

test['Net Profit to TL Ratio'] = test['Net Income']/test['Total Liabilities']

#Debt
# Cannot use Debt Service COverage ratio as proper metrics are not available

test['Debt to Equity Ratio'] = test['Total Liabilities']/test['Total Equity']

test['Debt to Asset Ratio'] = test['Total Liabilities']/test['Total Asset']

#EBITDA

test['EBITDA to TA Ratio'] = test['EBITDA']/test['Total Asset']

test['EBITDA to Equity Ratio'] = test['EBITDA']/test['Total Equity']

test['EBITDA to Sales Ratio'] = test['EBITDA']/test['Sales']

# Equity

test['Equity to TA Ratio'] = test['Total Equity']/test['Total Asset']

test['Equity to FA Ratio'] = test['Total Equity']/test['Total Fixed Asset']

test['Equity to TL Ratio'] = test['Total Equity']/test['Total Liabilities']

# Cash

test['Cash to Equity'] = test['Cash']/test['Total Equity']

test['Cash to TA'] = test['Cash']/test['Total Asset']

test['Cash to Sales'] = test['Cash']/test['Sales']


# In[31]:


test.dtypes


# In[32]:


# test.to_csv("test_trial.csv")


# In[33]:


test.replace([np.inf, -np.inf], np.nan, inplace=True)


# In[34]:


test.isnull().sum()


# In[35]:


colnames = list(test)
print(colnames)


# In[36]:


colnames = ['UEN',  'Revenue',  'Sales',  'other Rev',  'COGS',  'Gross Profit',  'total other Income',  'Total salary and expense',  'Total operating expense',  'Total Amortization and Depreciation',  'Total Financial Charges',  'Total Selling and Distribution expense',  'Tax',  'Net Income',  'Cash',  'Receivables',  'Inventories',  'Total Current Asset',  'Total Fixed Asset',  'Intangible Asset',  'Total Asset',  'Trade Payables',  'Term Loan',  'Hire Purchase Creditors',  'Income Tax Payables',  'Amount owing to directors/shareholders/related parties',  'Total Current Liabilities',  'Long Term Loan',  'Hire Purchase Creditor_Long',  'Total Liabilities',  'Paid Up Capital',  'Retained Earning',  'Total Equity',  'EBITDA',  'EBIT',  'Current Ratio',  'Cash Ratio',  'Gross Margin Ratio',  'Operating Margin Ratio',  'Net Profit Ratio',  'Net Profit to TA Ratio',  'Net Profit to TL Ratio',  'Debt to Equity Ratio',  'Debt to Asset Ratio',  'EBITDA to TA Ratio',  'EBITDA to Equity Ratio',  'EBITDA to Sales Ratio',  'Equity to TA Ratio',  'Equity to FA Ratio',  'Equity to TL Ratio',  'Cash to Equity',  'Cash to TA',  'Cash to Sales',  'Label']


# In[37]:


test_arr = test[colnames]


# In[38]:


test_arr.head()


# In[39]:


test_arr = test_arr.drop(['UEN'], axis=1)


# In[40]:


test_arr.to_csv('loan_test.csv', index = False)


# In[41]:


test_arr.head()


# In[42]:


len(test_arr.columns)


# In[ ]:




