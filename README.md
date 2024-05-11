# Hedging of Financial Derivative

Welcome to Hedging of Financial Derivative!

This project focuses on implementing a robust trading strategy using statistical arbitrage and convergence techniques for hedging financial derivatives.

## Overview

The project utilizes:
- Financial Programming
- Deep learning
- Machine learning

## Getting Started

To contribute to this project, follow these steps:

1. **Fork** the repository on GitHub.
2. **Clone** the forked project to your local machine: \`git clone <repository-URL>\`
3. Create a new branch for your work: \`git checkout -b your-branch-name\`
4. Make changes and improvements in your branch.
5. Commit your changes: \`git commit -m 'Add your descriptive commit message'\`
6. Push your changes to your forked repository: \`git push origin your-branch-name\`
7. Submit a **Pull Request** (PR) to the main repository for review.

## Ways to Contribute

We welcome contributions in various forms, such as:

- Reporting bugs or issues
- Providing feedback on the existing codebase
- Submitting fixes for identified issues
- Proposing new features or enhancements
- Improving documentation
- Adding code snippets, algorithms, or techniques related to financial programming

## Code Guidelines

Please adhere to proper coding standards and conventions:
- Follow clear and descriptive commit messages.
- Provide adequate comments within the code for readability.
- Thoroughly test your changes before submitting a PR.

## Issue Tracking

We use GitHub issues to manage tasks. Feel free to open an issue for bugs, suggestions, or discussions related to the project.

## Code of Conduct

We maintain a Code of Conduct to ensure a welcoming environment for all contributors. Please review and follow our [Code of Conduct](Code-of-conduct.md)

Thank you for your interest in contributing to the Financial Derivative Hedging Project!


# Hedging of Financial Derivative

Hedging is a market-neutral trading strategy that enables traders to profit from virtually any market conditions: uptrend, downtrend, or sideways movement. This strategy is categorized as a statistical arbitrage and convergence trading strategy.<br>
<br>
<ol>
  <li> To explain what our strategy exactly does and how is it working, we have taken the example of the Indian Market, we have taken a few securities over a specified time interval. Then we found all the cointegrated pairs of stocks from all the possible pairs of securities by considering all the stocks having p-values less than a certain cut-off. For further analysis, we have identified the security pair with minimum p-value.</li>
  <li> Next we have calculated the spread of the two series. To calculate the spread, we use linear regression to get the coefficient for the linear combination to construct between our two securities. Since the absolute spread isn't very useful in statistical terms. It is more helpful to normalize our signal by treating it as a z-score. A Z-score is nothing but a numerical measurement that describes a value's relationship to the mean of a group of values. The Z-score is measured in terms of standard deviations from the mean.</li>
  <li>Next we define our strategy and generate the trading signals during backtesting on another period which we have defined-
   <ul>
    <li>Go "Long" the spread whenever the z-score is below -1.0</li>
    <li>Go "Short" the spread when the z-score is above 1.0</li>
    <li>Exit positions when the z-score approaches zero</li>
   </ul>
   <li>Next we have created and calculated the returns of our portfolio based on strategy.</li>
</ol>
