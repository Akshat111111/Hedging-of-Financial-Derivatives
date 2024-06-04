# Stock Price Forecasting Project

## Requirements
The project aims to forecast stock prices for Google, Microsoft, and Apple using historical data from Yahoo Finance. Key requirements include:
- Fetching historical stock price data from Yahoo Finance.
- Building time series forecasting models using the Prophet library.
- Visualizing historical and forecasted stock prices.
- Allowing user interaction to input specific dates or time periods for estimated stock prices.

## Approach
1. **Data Retrieval:** Use the `yfinance` library to fetch historical stock price data for the three companies.
2. **Model Building:** Utilize the `Prophet` library to build time series forecasting models for each company's stock prices.
3. **Visualization:** Visualize the historical and forecasted stock prices using `matplotlib`.
4. **User Interaction:** Implement user interaction features to input specific dates or time periods and obtain estimated stock prices.

## Explanations
- **Data Retrieval:** Historical stock price data is fetched from Yahoo Finance for the desired time period.
- **Model Building:** Time series forecasting models are built using the Prophet library, which is well-suited for handling seasonal and trend-based data.
- **Visualization:** Historical stock prices and forecasted prices are visualized on plots using Matplotlib, providing insights into trends and patterns.
- **User Interaction:** Users can interact with the system by inputting specific dates or time periods to obtain estimated stock prices, enhancing usability and decision-making capabilities.


