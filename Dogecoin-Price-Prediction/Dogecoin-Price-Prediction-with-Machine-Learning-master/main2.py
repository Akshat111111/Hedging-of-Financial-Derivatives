import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import regression
sns.set()
plt.style.use('seaborn-whitegrid')
data = pd.read_csv("AAPL.csv")

data = data[["Date", "Close"]]
data["Date"] = pd.to_datetime(data.Date)
data["Close"].plot(figsize=(12, 8), title="Apple Stock Prices", fontsize=20, label="Close Price")
plt.legend()
plt.grid()
plt.show()

