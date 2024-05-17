import numpy as np
import statsmodels.tsa.api as sm

def garch_model_11(asset_returns):
  """
  Fits a GARCH(1,1) model to asset returns (basic implementation).

  Args:
      asset_returns: A NumPy array of daily asset returns.

  Returns:
      The fitted GARCH(1,1) model.
  """
  model = sm.tsa.garch.GARCH(asset_returns, order=(1, 1))
  model_fit = model.fit()
  return model_fit
