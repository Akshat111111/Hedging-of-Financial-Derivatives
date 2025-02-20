import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gradio as gr

# Load the CSV file
data_path = 'uae_properties.csv'

def load_data():
    return pd.read_csv(data_path)

def visualize_data(max_price):
    # Load data
    df = load_data()

    # Filter data based on max price
    df['price'] = df.apply(lambda row: row['price'] * 100 if row['price_unit'] == 'Cr' else row['price'], axis=1)
    filtered_df = df[df['price'] <= max_price]

    if filtered_df.empty:
        return "Error: No data available for the selected price range.", None

    # Plot bar chart for average property prices by locality
    plt.figure(figsize=(10, 6))
    sns.barplot(x='locality', y='price', data=filtered_df, ci=None)
    plt.xticks(rotation=90)
    plt.title('Average Property Prices by Locality')
    plt.xlabel('Locality')
    plt.ylabel('Average Price (in L)')
    plt.tight_layout()

    # Save plot to a file
    plot_path = 'bar_plot.png'
    plt.savefig(plot_path)
    plt.close()

    # Generate EDA summary
    eda_summary = filtered_df.describe(include='all').to_string()

    return eda_summary, plot_path

iface = gr.Interface(
    fn=visualize_data,
    inputs=gr.Number(label="Max Price (in L)"),
    outputs=[gr.Textbox(label="EDA Summary"), gr.Image(type="filepath", label="Bar Plot")]
)

iface.launch(share=True)
