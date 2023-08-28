# Hedging of Financial Derivative

A Hedging is a market neutral trading strategy enabling traders to profit from virtually any market conditions: uptrend, downtrend, or sideways movement. This strategy is categorized as a statistical arbitrage and convergence trading strategy.<br>
<br>
1) To explain what our strategy exactly does and how is it working, we have taken the example of the Indian Market, we have taken a few securities over a specified time interval. Then we have found all the cointegrated pairs of stocks from all the possible pairs of securities by considering all the stocks having p-value less than a certain cut-off. For further analysis, we have identified the security pair with minimum p-value.<br>
2) Next we have calculated the spread of the two series. In order to actually calculate the spread, we use a linear regression to get the coefficient for the linear combination to construct between our two securities. Since, the absolute spread isn't very useful in statistical terms. It is more helpful to normalize our signal by treating it as a z-score. A Z-score is nothing but a numerical measurement that describes a value's relationship to the mean of a group of values. Z-score is measured in terms of standard deviations from the mean.<br>
3) Next we define our strategy and generate the trading signals during backtesting on another time period which we have defined-<br>
   * Go "Long" the spread whenever the z-score is below -1.0
   * Go "Short" the spread when the z-score is above 1.0
   * Exit positions when the z-score approaches zero
4)Next we have created calculated the returns of our portfolio on the basis of strategy
