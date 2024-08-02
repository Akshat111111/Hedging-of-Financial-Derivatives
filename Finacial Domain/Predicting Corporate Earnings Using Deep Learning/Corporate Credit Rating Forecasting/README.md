# Corporate Credit Rating Forecasting

This repository contains the results of a data analysis performed on a set of corporate credit ratings given by ratings agencies to a set of companies. The aim of the data analysis is to build a machine learning model from the rating data that can be used to predict the rating a company will receive.

## The Dataset

The dataset was generated with the file `generateCreditRatingDataset.py`. It makes use of a api and a previous dataset. More in the acknowledgement session.

There are 30 features for every company of which 25 are financial indicators. They can be divided in:

- `Liquidity Measurement Ratios:` currentRatio, quickRatio, cashRatio, daysOfSalesOutstanding
- `Profitability Indicator Ratios:` grossProfitMargin, operatingProfitMargin, pretaxProfitMargin, netProfitMargin, effectiveTaxRate, returnOnAssets, returnOnEquity, returnOnCapitalEmployed
- `Debt Ratios:` debtRatio, debtEquityRatio
- `Operating Performance Ratios:` assetTurnover
- `Cash Flow Indicator Ratios`: operatingCashFlowPerShare, freeCashFlowPerShare, cashPerShare, operatingCashFlowSalesRatio, freeCashFlowOperatingCashFlowRatio


## Results

We achieve an accuracy of 69.14% with an XGboost model.


## Companies

We can see companies such as `Walt Disney` and `Philip Morris` are low risk. `Foot locker` and `MGM` are considered risky companies.   



## Acknowledgement

Sorces of Data: Thanks a lot for these services and their amazing datasets.
`Credit Rating:` [opendatasoft](https://public.opendatasoft.com/)
`Financial Informatino:` [financialmodelingprep](https://financialmodelingprep.com/)