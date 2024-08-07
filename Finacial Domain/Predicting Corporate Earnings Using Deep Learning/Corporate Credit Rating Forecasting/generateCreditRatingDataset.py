# By Alan Gewerc

"""
Generating a Corporate Credit Rating Dataset for Classification
The dataset is composed by Financials + Credit Rating

Sources of data:
https://public.opendatasoft.com/
https://financialmodelingprep.com/
"""

# Import Libraries
import pandas as pd
import re
import json
from urllib.request import urlopen
import os

path = 'C:/Users/hp/OneDrive/Content/Blog/Finance/Github' 
os.chdir(path)


# Import Data
df_is_stocks = pd.read_csv('data/us_stocks.csv')
df_ratings = pd.read_csv('data/ratings-history.csv')

# Drop NA
df_ratings = df_ratings.dropna()

# Drop Duplicates
df_ratings = df_ratings.drop_duplicates()

# Standarize ratings
def standarizeRating(name):
    
    name = re.sub(r'Aaa','AAA', name)
    name = re.sub(r'Aa1', 'AA+', name)
    name = re.sub(r'Aa2', 'AA', name)
    name = re.sub(r'Aa3	', 'AA−', name)
    name = re.sub(r'A1', 'A+', name)    
    name = re.sub(r'A2','A', name)   
    name = re.sub(r'A3', 'A-', name)
    name = re.sub(r'Baa1','BBB+', name)
    name = re.sub(r'Baa2','BBB', name)
    name = re.sub(r'Baa3', 'BBB-', name)
    name = re.sub(r'Ba1','BB+', name)
    name = re.sub(r'Ba2', 'BB', name)
    name = re.sub(r'Ba3', 'BB−', name)
    name = re.sub(r'B1','B+', name)    
    name = re.sub(r'B2','B', name)   
    name = re.sub(r'B3','B−', name)
    name = re.sub(r'Caa1','CCC+', name)
    name = re.sub(r'Caa2','CCC', name)
    name = re.sub(r'Caa3', 'CCC−', name)
    return name

# Apply transformation
df_ratings['Rating'] = df_ratings['Rating'].apply(str).apply(standarizeRating)

# We do not wan't to consider if a Rating was A+ or A. 
# We remove  + and - from ratings
df_ratings['Rating'] = df_ratings['Rating'].str.replace(r'+', '').str.replace(r'-', '')


# Filter Ratings
# Exclude non-traditional ratings from the datasets
rating_set = {'A', 'AA', 'AAA', 
              'B', 'BB', 'BBB', 
              'C', 'CC', 'CCC', 
              'D'}

df_ratings = df_ratings[df_ratings['Rating'].isin(rating_set)]

# Filter duplicates in the dataset
df_ratings = df_ratings.drop_duplicates(subset = df_ratings.columns)

df_ratings.groupby('Rating').count()




# Clean company names
# Remove unwated characters from the names
# We want to merge the datasets df_is_stocks and df_ratings
# The best method is to remove all possible characters that will cause mismatch
# This function will remove all characters and names we don't want


def clean(name):
    
    name = name.lower()
    name = re.sub(r'\.', '', name)
    name = re.sub(r',', '', name)
    name = re.sub(r'^a-z', '', name)
    name = re.sub(r'corporation', '', name)
    name = re.sub(r' corp', '', name)    
    name = re.sub(r' co', '', name)   
    name = re.sub(r'inc', '', name)
    name = re.sub(r'limited', '', name)
    name = re.sub(r'ltd', '', name)
    name = re.sub(r'holdings', '', name)
    name = re.sub(r' holding', '', name)    
    name = re.sub(r'plc', '', name)
    name = re.sub(r'group', '', name)
    name = re.sub(r' ag', '', name)
    name = re.sub(r' sa', '', name)
    name = re.sub(r' pty', '', name)
    name = re.sub(r' international', '', name)
    name = re.sub(r' incorporated', '', name)
    name = re.sub(r' spa', '', name)
    name = re.sub(r' se', '', name)
    name = re.sub(r' lp', '', name)    
    name = re.sub(r' (The)', '', name)    
    name = re.sub(r'The', '', name)    
    name = re.sub(r'LLC', '', name)   
    name = re.sub(r'n.v', '', name)   
    name = name.strip()
    
    return name


# Apply the function in both datasets
df_is_stocks.loc[:, 'Clean_Name'] = df_is_stocks['Name']
df_is_stocks.loc[:, 'Clean_Name'] = df_is_stocks['Name'].apply(clean)
df_ratings.loc[:, 'Clean_Name'] = df_ratings['Name']
df_ratings.loc[:, 'Clean_Name'] = df_ratings['Clean_Name'].apply(str).apply(clean)

# Merge Datasets
df_ratings = pd.merge(df_ratings, df_is_stocks ,on='Clean_Name')



# Clean the table from the merge
df_ratings = df_ratings.rename(columns = {'Name_x': 'Name'}).drop('Name_y', axis = 1)

# Visualise the data
df_ratings.groupby('Rating').count()

df_ratings = df_ratings.drop_duplicates(subset = ['Rating Action Date', 'Clean_Name'])


####################################################
####################################################
####################################################
####################################################
# Now we will get the financial
# We will use the API Financial Modeling Prep
# Create function to get data

def get_jsonparsed_data(url):
    
    response = urlopen(url)
    data = response.read().decode("utf-8")
    return json.loads(data)


# List of the symbols/companies 
Symbol_list = list(set(df_ratings.Symbol))

# Remove Unwated characters
Symbol_list = [re.sub(r'\^.+', '', item) for item in Symbol_list]

# Create a dataframe to place the the data
df_company_ratios = pd.DataFrame.from_dict(
        get_jsonparsed_data("https://financialmodelingprep.com/api/v3/ratios/" 
                            + Symbol_list[0] + 
                            "?apikey=c283e7eb3ddc10dcc0ed8146046dff97"))

df_company_ratios = df_company_ratios[0:0] # empty the dataframe

# Get every company from the list the financial information
for Symbol in Symbol_list:
    
    if Symbol.isalpha() == True:
    
        ratios = pd.DataFrame.from_dict(
                get_jsonparsed_data("https://financialmodelingprep.com/api/v3/ratios/" + 
                                    Symbol + 
                                    "?apikey=c283e7eb3ddc10dcc0ed8146046dff97"))
    
        frames = [df_company_ratios, ratios]
    
        df_company_ratios = pd.concat(frames)   



# Now filter by date
# The api returns every available information from the past
# we want exclusively the data from the year of the rating or the year before
# If the rating was done before june we will take the indicators from the year before
# If it was after june we will take from the following year
df_ratings = df_ratings.rename(columns = {'Rating Action Date':'date'})
df_ratings['date'] = df_ratings['date'].astype('datetime64[ns]')
df_ratings['month_rating'] = df_ratings['date'].dt.month
df_ratings['year_change'] = [1 if x < 7 else 0 for x in df_ratings['month_rating']]
df_ratings['year'] = df_ratings['date'].dt.year
df_ratings['Year'] = df_ratings['year'] - df_ratings['year_change']

df_company_ratios = df_company_ratios.rename(columns = {'symbol':'Symbol'})
df_company_ratios['date'] = df_company_ratios['date'].astype('datetime64[ns]')
df_company_ratios['Year'] = df_company_ratios['date'].dt.year


# merge Datasets
df_ratings = pd.merge(df_ratings, df_company_ratios, on=['Symbol', 'Year']) 

# Work on missing values and eliminate duplicates
a = df_ratings.count()
the_list = a[a<5900].reset_index()['index']
df_ratings_NA = df_ratings[df_ratings.columns.difference(the_list)]
df_ratings_NA = df_ratings_NA.dropna()
df_ratings_NA = df_ratings_NA.drop_duplicates(['Clean_Name', 'date_y'])


# clean the dataset - drop columns
df_ratings_NA = df_ratings_NA.drop(['Name', 'Year', 'date_y', 'ebtPerEbit',
                                    'month_rating', 'netIncomePerEBT', 'year',
                                    'year_change', 'industry'
                                    'priceCashFlowRatio', 'priceEarningsRatio',	
                                    'priceSalesRatio',
                                    'priceToFreeCashFlowsRatio',	
                                    'priceToOperatingCashFlowsRatio',	
                                    'priceToSalesRatio'], axis=1)


# clean the dataset - Rename columns
df_ratings_NA = df_ratings_NA.rename({'Clean_Name' :'Name', 'date_x':'Date'})

# Export to CSVDatasets
df_ratings_NA.to_csv (r'data/rating_dataframe.csv', index = False, header=True)






