#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import datetime
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt  
import seaborn as sns



pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows',500)



loan = pd.read_excel('train_data.xlsx', sheet_name = 'Raw_data')
loan


# ### Checking data type
# #### Identifying discrepancies and standardising data types
loan.dtypes
invalid_rows = loan['Y1_Total Selling and Distribution expense'].apply(lambda x: isinstance(x, str))
row_inv = loan[invalid_rows].index
print(row_inv)

loan.iloc[82]

#After evaluation of the row, the Net profit is not affected by the incorrect figure from the OCR, hence it is
#made NaN
loan.loc[invalid_rows, 'Y1_Total Selling and Distribution expense'] = np.nan


loan['Y1_Total Selling and Distribution expense'] = loan['Y1_Total Selling and Distribution expense'].apply(pd.to_numeric, errors='coerce')

# The data types are all float and accurate in accordance with the dataset
loan.dtypes


# ### Data Cleaning as per Notes provided
# #### 1. Filtering out application dates that are older than 2018
# #### 2. Removing duplicates

#Filtering application dates older than 2018
loan = loan[loan['Application Date'].dt.strftime('%Y') >= '2018']


len(loan)


loan = loan.sort_values(['Application Date','Financial as of']).drop_duplicates(['UEN'], keep='last')


len(loan)


loan.isnull().sum()


# ### Removing redundant columns, replacing NaN with 0 for specific columns and dropping NaN

#After studying the dataset, sales is equal to revenue in most cases and since there are only 3 missing data point
#in sales, it can be assumed that sales = revenue

loan['Y1_Sales'] = loan['Y1_Sales'].fillna(loan['Y1_Revenue'])


# Removing columns that do not add up to totals and are not requisite for calculations
# While it would be great to assess the inventory turnover ratio in terms of loan qualification.  
# The column has 83 zeroes. Additionally, it has 28 NaN which if replaced with zero, would result in 111 entries with 
# zeroes. That would account to ~65% of the data giving little to no value.

loan = loan.drop(['Application Date','Financial as of'], axis=1)


loan.isnull().sum()

cols_to_convert = ['Y1_COGS','Y1_total other Income', 'Y1_Total salary and expense', 'Y1_Total operating expense', 'Y1_Total Amortization and Depreciation', 'Y1_Total Financial Charges', 'Y1_Total Selling and Distribution expense', 'Y1_Tax', 'Y1_Cash', 'Y1_Intangible Asset','Y1_Inventories','Y1_Total Fixed Asset','Y1_Trade Payables', 'Y1_Term Loan', 'Y1_Hire Purchase Creditors', 'Y1_Income Tax Payables', 'Y1_Amount owing to directors/shareholders/related parties', 'Y1_Long Term Loan', 'Y1_Hire Purchase Creditor_Long']
loan[cols_to_convert] = loan[cols_to_convert].replace(np.nan, 0)

loan['Y1_Total Current Asset'] = loan['Y1_Total Current Asset'].fillna(loan['Y1_Cash'] + loan['Y1_Inventories'] + loan['Y1_Receivables'])

loan['Y1_Total Current Liabilities'] = loan['Y1_Total Current Liabilities'].fillna(loan['Y1_Term Loan'] + loan['Y1_Hire Purchase Creditors'] + loan['Y1_Trade Payables'] + loan['Y1_Income Tax Payables'] + loan['Y1_Amount owing to directors/shareholders/related parties'])

loan['Y1_Total Equity'] = loan['Y1_Total Equity'].fillna(loan['Y1_Retained Earning'] + loan['Y1_Paid Up Capital'])

loan.isnull().sum()

loan = loan.dropna()

len(loan)

loan.columns = loan.columns.str.lstrip('Y1_')

loan

# ### Creating the different financial ratios

# #### Adding EBITDA and EBIT in the dataframe

loan['EBITDA'] = loan['Net Income'] + loan['Tax'] + loan['Total Amortization and Depreciation']

loan['EBIT'] = loan['Net Income'] + loan['Tax']

# loan['Total Revenue'] = loan['Sales'] + loan['other Rev']


# #### Creating the Ratios


#Liquidity Ratios
loan['Current Ratio'] = loan['Total Current Asset']/loan['Total Current Liabilities']

loan['Cash Ratio'] = loan['Cash']/loan['Total Current Liabilities']

#Profitability Ratios
loan['Gross Margin Ratio'] = loan['Gross Profit']/loan['Sales']

loan['Operating Margin Ratio'] = loan['EBIT']/loan['Revenue']

loan['Net Profit Ratio'] = loan['Net Income']/loan['Sales']

loan['Net Profit to TA Ratio'] = loan['Net Income']/loan['Total Asset']

loan['Net Profit to TL Ratio'] = loan['Net Income']/loan['Total Liabilities']

#Debt
# Cannot use Debt Service COverage ratio as proper metrics are not available

loan['Debt to Equity Ratio'] = loan['Total Liabilities']/loan['Total Equity']

loan['Debt to Asset Ratio'] = loan['Total Liabilities']/loan['Total Asset']

#EBITDA

loan['EBITDA to TA Ratio'] = loan['EBITDA']/loan['Total Asset']

loan['EBITDA to Equity Ratio'] = loan['EBITDA']/loan['Total Equity']

loan['EBITDA to Sales Ratio'] = loan['EBITDA']/loan['Sales']

# Equity

loan['Equity to TA Ratio'] = loan['Total Equity']/loan['Total Asset']

loan['Equity to FA Ratio'] = loan['Total Equity']/loan['Total Fixed Asset']

loan['Equity to TL Ratio'] = loan['Total Equity']/loan['Total Liabilities']

# Cash

loan['Cash to Equity'] = loan['Cash']/loan['Total Equity']

loan['Cash to TA'] = loan['Cash']/loan['Total Asset']

loan['Cash to Sales'] = loan['Cash']/loan['Sales']


loan


loan.dtypes


# loan.to_csv('loan_train.csv', index = False)


loan.replace([np.inf, -np.inf], np.nan, inplace=True)
# loan.to_csv('loan_train.csv', index = False)


loan.isnull().sum()

ratios_to_convert = ['Current Ratio', 'Cash Ratio', 'Net Profit to TA Ratio', 'Net Profit to TL Ratio', 'Debt to Equity Ratio', 'Debt to Asset Ratio', 'EBITDA to TA Ratio','EBITDA to Equity Ratio','Equity to TA Ratio', 'Equity to FA Ratio', 'Equity to TL Ratio', 'Cash to Equity', 'Cash to TA']
loan[ratios_to_convert] = loan[ratios_to_convert].replace(np.nan, 0)


loan.isnull().sum()


colnames = list(loan)

print(colnames)
colnames = ['UEN',  'Revenue',  'Sales',  'other Rev',  'COGS',  'Gross Profit',  'total other Income',  'Total salary and expense',  'Total operating expense',  'Total Amortization and Depreciation',  'Total Financial Charges',  'Total Selling and Distribution expense',  'Tax',  'Net Income',  'Cash',  'Receivables',  'Inventories',  'Total Current Asset',  'Total Fixed Asset',  'Intangible Asset',  'Total Asset',  'Trade Payables',  'Term Loan',  'Hire Purchase Creditors',  'Income Tax Payables',  'Amount owing to directors/shareholders/related parties',  'Total Current Liabilities',  'Long Term Loan',  'Hire Purchase Creditor_Long',  'Total Liabilities',  'Paid Up Capital',  'Retained Earning',  'Total Equity',  'EBITDA',  'EBIT',  'Current Ratio',  'Cash Ratio',  'Gross Margin Ratio',  'Operating Margin Ratio',  'Net Profit Ratio',  'Net Profit to TA Ratio',  'Net Profit to TL Ratio',  'Debt to Equity Ratio',  'Debt to Asset Ratio',  'EBITDA to TA Ratio',  'EBITDA to Equity Ratio',  'EBITDA to Sales Ratio',  'Equity to TA Ratio',  'Equity to FA Ratio',  'Equity to TL Ratio',  'Cash to Equity',  'Cash to TA',  'Cash to Sales',  'Label']

loan_arr = loan[colnames]

loan_arr.head()


loan_arr = loan_arr.drop(['UEN'], axis=1)

loan_arr.to_excel('Cleaned.xlsx', index = False)

len(loan_arr)

# ### EDA
# ### Checking distribution of continuous variables


#Work on this

sns.set(style = 'white', font = 'Georgia')

fig = plt.figure(figsize=(25,20))

i = 1

for column in loan.columns:
    plt.subplot(4,5,i)
    fig.subplots_adjust(hspace = 0.3, wspace = 0.3)
    chart = sns.histplot(loan[column], kde = True, kde_kws=dict(cut=3))
    chart.set_xlabel(chart.get_xlabel(), fontsize = 14, fontweight = 'bold')
    chart.set_ylabel(chart.get_ylabel(), fontsize = 14, fontweight = 'bold')
    chart.tick_params(axis = 'both', which = 'major', labelsize = 12)
    
    i+=1


# ### Checking for Correlation


plt.figure(figsize = (45,30))
hm = sns.heatmap(loan.corr(), annot = True, cmap = "coolwarm")
hm.set(xlabel = '\nCompany Details', ylabel='Company Details', title = "Correlation matrix of  dataset\n")
plt.show()


# ### Calculating Variance Inflation Factor


# Check on inf and NaNs for the ratios calculated - resolve

df2 = pd.DataFrame()

for column in loan.select_dtypes(exclude=['object', 'int']):
    df2[column] = loan[column]

vif_data = pd.DataFrame()
vif_data["feature"] = df2.columns

# calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(df2.values, i)
                   for i in range(len(df2.columns))]

vif_data.sort_values(by = "VIF", ascending = False)


loan.notnull().sum()


loan.isnull().sum()

count = np.isinf(loan).values.sum()
count



# ### Splitting the data - train and validation 
# ### Running three models -  Logistic Regression, Random Forest, Neural Network
# ### Model Comparison - the metrics are R squared, MAPE, RMSE, ROC/AUC
