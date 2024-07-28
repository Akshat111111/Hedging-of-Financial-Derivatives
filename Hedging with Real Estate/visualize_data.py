import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def plot_data(df):
    sns.pairplot(df)
    plt.show()
if __name__ == "__main__":
    df = pd.read_csv('path/to/data.csv')
    plot_data(df)
