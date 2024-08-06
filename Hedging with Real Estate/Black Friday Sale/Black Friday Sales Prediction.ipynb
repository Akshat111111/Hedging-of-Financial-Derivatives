{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Problem Statement"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "A retail company wants to understand the customer purchase behaviour (specifically, purchase amount) against various products of different categories. They have shared purchase summary of various customers for selected high volume products from last month.\n",
    "The data set also contains customer demographics (age, gender, marital status, city_type, stay_in_current_city), product details (product_id and product category) and Total purchase_amount from last month.\n",
    "\n",
    "Now, they want to build a model to predict the purchase amount of customer against various products which will help them to create personalized offer for customers against different products."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Variable\tDefinition"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 124,
   "metadata": {},
   "outputs": [],
   "source": [
    "# User_ID\tUser ID\n",
    "# Product_ID\tProduct ID\n",
    "# Gender\tSex of User\n",
    "# Age\tAge in bins\n",
    "# Occupation\tOccupation (Masked)\n",
    "# City_Category\tCategory of the City (A,B,C)\n",
    "# Stay_In_Current_City_Years\tNumber of years stay in current city\n",
    "# Marital_Status\tMarital Status\n",
    "# Product_Category_1\tProduct Category (Masked)\n",
    "# Product_Category_2\tProduct may belongs to other category also (Masked)\n",
    "# Product_Category_3\tProduct may belongs to other category also (Masked)\n",
    "# Purchase\tPurchase Amount (Target Variable)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Importing Libraries and Loading data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 125,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 126,
   "metadata": {},
   "outputs": [],
   "source": [
    "data = pd.read_csv(\"https://raw.githubusercontent.com/nanthasnk/Black-Friday-Sales-Prediction/master/Data/BlackFridaySales.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 127,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>User_ID</th>\n",
       "      <th>Product_ID</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Occupation</th>\n",
       "      <th>City_Category</th>\n",
       "      <th>Stay_In_Current_City_Years</th>\n",
       "      <th>Marital_Status</th>\n",
       "      <th>Product_Category_1</th>\n",
       "      <th>Product_Category_2</th>\n",
       "      <th>Product_Category_3</th>\n",
       "      <th>Purchase</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>1000001</td>\n",
       "      <td>P00069042</td>\n",
       "      <td>F</td>\n",
       "      <td>0-17</td>\n",
       "      <td>10</td>\n",
       "      <td>A</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>8370</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>1000001</td>\n",
       "      <td>P00248942</td>\n",
       "      <td>F</td>\n",
       "      <td>0-17</td>\n",
       "      <td>10</td>\n",
       "      <td>A</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>6.0</td>\n",
       "      <td>14.0</td>\n",
       "      <td>15200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>1000001</td>\n",
       "      <td>P00087842</td>\n",
       "      <td>F</td>\n",
       "      <td>0-17</td>\n",
       "      <td>10</td>\n",
       "      <td>A</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>12</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1422</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>1000001</td>\n",
       "      <td>P00085442</td>\n",
       "      <td>F</td>\n",
       "      <td>0-17</td>\n",
       "      <td>10</td>\n",
       "      <td>A</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>12</td>\n",
       "      <td>14.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1057</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>1000002</td>\n",
       "      <td>P00285442</td>\n",
       "      <td>M</td>\n",
       "      <td>55+</td>\n",
       "      <td>16</td>\n",
       "      <td>C</td>\n",
       "      <td>4+</td>\n",
       "      <td>0</td>\n",
       "      <td>8</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>7969</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   User_ID Product_ID Gender   Age  Occupation City_Category  \\\n",
       "0  1000001  P00069042      F  0-17          10             A   \n",
       "1  1000001  P00248942      F  0-17          10             A   \n",
       "2  1000001  P00087842      F  0-17          10             A   \n",
       "3  1000001  P00085442      F  0-17          10             A   \n",
       "4  1000002  P00285442      M   55+          16             C   \n",
       "\n",
       "  Stay_In_Current_City_Years  Marital_Status  Product_Category_1  \\\n",
       "0                          2               0                   3   \n",
       "1                          2               0                   1   \n",
       "2                          2               0                  12   \n",
       "3                          2               0                  12   \n",
       "4                         4+               0                   8   \n",
       "\n",
       "   Product_Category_2  Product_Category_3  Purchase  \n",
       "0                 NaN                 NaN      8370  \n",
       "1                 6.0                14.0     15200  \n",
       "2                 NaN                 NaN      1422  \n",
       "3                14.0                 NaN      1057  \n",
       "4                 NaN                 NaN      7969  "
      ]
     },
     "execution_count": 127,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 128,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(537577, 12)"
      ]
     },
     "execution_count": 128,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 129,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 537577 entries, 0 to 537576\n",
      "Data columns (total 12 columns):\n",
      "User_ID                       537577 non-null int64\n",
      "Product_ID                    537577 non-null object\n",
      "Gender                        537577 non-null object\n",
      "Age                           537577 non-null object\n",
      "Occupation                    537577 non-null int64\n",
      "City_Category                 537577 non-null object\n",
      "Stay_In_Current_City_Years    537577 non-null object\n",
      "Marital_Status                537577 non-null int64\n",
      "Product_Category_1            537577 non-null int64\n",
      "Product_Category_2            370591 non-null float64\n",
      "Product_Category_3            164278 non-null float64\n",
      "Purchase                      537577 non-null int64\n",
      "dtypes: float64(2), int64(5), object(5)\n",
      "memory usage: 49.2+ MB\n"
     ]
    }
   ],
   "source": [
    "data.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "`Age` should be treated as a numerical column\n",
    "\n",
    "`City_Category` we can convert this to a numerical column and should look at the frequency of each city category.\n",
    "\n",
    "`Gender` has two values and should be converted to binary values\n",
    "\n",
    "`Product_Category_2` and `Product_Category_3` have null values"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Checking Null values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 130,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "User_ID                            0\n",
       "Product_ID                         0\n",
       "Gender                             0\n",
       "Age                                0\n",
       "Occupation                         0\n",
       "City_Category                      0\n",
       "Stay_In_Current_City_Years         0\n",
       "Marital_Status                     0\n",
       "Product_Category_1                 0\n",
       "Product_Category_2            166986\n",
       "Product_Category_3            373299\n",
       "Purchase                           0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 130,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Null Value in percentage"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 131,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "User_ID                        0.000000\n",
       "Product_ID                     0.000000\n",
       "Gender                         0.000000\n",
       "Age                            0.000000\n",
       "Occupation                     0.000000\n",
       "City_Category                  0.000000\n",
       "Stay_In_Current_City_Years     0.000000\n",
       "Marital_Status                 0.000000\n",
       "Product_Category_1             0.000000\n",
       "Product_Category_2            31.062713\n",
       "Product_Category_3            69.441029\n",
       "Purchase                       0.000000\n",
       "dtype: float64"
      ]
     },
     "execution_count": 131,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.isnull().sum()/data.shape[0]*100"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "There are 31% null values in the `Product_Category_2` and 69% null values in the `Product_Category_3`"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Unique elements in each attributes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 132,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "User_ID                        5891\n",
       "Product_ID                     3623\n",
       "Gender                            2\n",
       "Age                               7\n",
       "Occupation                       21\n",
       "City_Category                     3\n",
       "Stay_In_Current_City_Years        5\n",
       "Marital_Status                    2\n",
       "Product_Category_1               18\n",
       "Product_Category_2               17\n",
       "Product_Category_3               15\n",
       "Purchase                      17959\n",
       "dtype: int64"
      ]
     },
     "execution_count": 132,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.nunique()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can drop `User_ID` and `Product_ID` for model prediction as it has more unique values."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# EDA"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Target Variable Purchase"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 133,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.distplot(data[\"Purchase\"],color='r')\n",
    "plt.title(\"Purchase Distribution\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We can observe that purchase amount is repeating for many customers.This may be because on Black Friday many are buying discounted products in large numbers and kind of follows a Gaussian Distribution."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 134,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.boxplot(data[\"Purchase\"])\n",
    "plt.title(\"Boxplot of Purchase\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 135,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.6242797316083074"
      ]
     },
     "execution_count": 135,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data[\"Purchase\"].skew()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 136,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "-0.34312137256836284"
      ]
     },
     "execution_count": 136,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data[\"Purchase\"].kurtosis()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 137,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "count    537577.000000\n",
       "mean       9333.859853\n",
       "std        4981.022133\n",
       "min         185.000000\n",
       "25%        5866.000000\n",
       "50%        8062.000000\n",
       "75%       12073.000000\n",
       "max       23961.000000\n",
       "Name: Purchase, dtype: float64"
      ]
     },
     "execution_count": 137,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data[\"Purchase\"].describe()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The purchase is right skewed and we can observe multiple peaks in the distribution we can do a log transformation for the purchase."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Gender"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 138,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(data['Gender'])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 139,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "M    75.408732\n",
       "F    24.591268\n",
       "Name: Gender, dtype: float64"
      ]
     },
     "execution_count": 139,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data['Gender'].value_counts(normalize=True)*100"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "There are more males than females"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 140,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Gender\n",
       "F    8809.761349\n",
       "M    9504.771713\n",
       "Name: Purchase, dtype: float64"
      ]
     },
     "execution_count": 140,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.groupby(\"Gender\").mean()[\"Purchase\"]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "On average the male gender spends more money on purchase contrary to female, and it is possible to also observe this trend by adding the total value of purchase."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Marital Status"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 141,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(data['Marital_Status'])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "There are more unmarried people in the dataset who purchase more"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 142,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Marital_Status\n",
       "0    9333.325467\n",
       "1    9334.632681\n",
       "Name: Purchase, dtype: float64"
      ]
     },
     "execution_count": 142,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.groupby(\"Marital_Status\").mean()[\"Purchase\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 143,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "data.groupby(\"Marital_Status\").mean()[\"Purchase\"].plot(kind='bar')\n",
    "plt.title(\"Marital_Status and Purchase Analysis\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This is interesting though unmarried people spend more on purchasing, the average purchase amount of married and unmarried people are the same."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Occupation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 144,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1296x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(18,5))\n",
    "sns.countplot(data['Occupation'])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Occupation has at least 20 different values. Since we do not known to each occupation each number corresponds, is difficult to make any analysis. Furthermore, it seems we have no alternative but to use since there is no way to reduce this number"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 145,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Purchase</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Occupation</th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>9186.946726</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>9017.703095</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>9025.938982</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>9238.077277</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>9279.026742</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>5</td>\n",
       "      <td>9388.848978</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>6</td>\n",
       "      <td>9336.378620</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>7</td>\n",
       "      <td>9502.175276</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>8</td>\n",
       "      <td>9576.508530</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>9</td>\n",
       "      <td>8714.335934</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>10</td>\n",
       "      <td>9052.836410</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>11</td>\n",
       "      <td>9299.467190</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>12</td>\n",
       "      <td>9883.052460</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>13</td>\n",
       "      <td>9424.449391</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>14</td>\n",
       "      <td>9568.536426</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>15</td>\n",
       "      <td>9866.239925</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>16</td>\n",
       "      <td>9457.133118</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>17</td>\n",
       "      <td>9906.378997</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>18</td>\n",
       "      <td>9233.671418</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>19</td>\n",
       "      <td>8754.249162</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>20</td>\n",
       "      <td>8881.099514</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "               Purchase\n",
       "Occupation             \n",
       "0           9186.946726\n",
       "1           9017.703095\n",
       "2           9025.938982\n",
       "3           9238.077277\n",
       "4           9279.026742\n",
       "5           9388.848978\n",
       "6           9336.378620\n",
       "7           9502.175276\n",
       "8           9576.508530\n",
       "9           8714.335934\n",
       "10          9052.836410\n",
       "11          9299.467190\n",
       "12          9883.052460\n",
       "13          9424.449391\n",
       "14          9568.536426\n",
       "15          9866.239925\n",
       "16          9457.133118\n",
       "17          9906.378997\n",
       "18          9233.671418\n",
       "19          8754.249162\n",
       "20          8881.099514"
      ]
     },
     "execution_count": 145,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "occup = pd.DataFrame(data.groupby(\"Occupation\").mean()[\"Purchase\"])\n",
    "occup"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 146,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1080x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "occup.plot(kind='bar',figsize=(15,5))\n",
    "plt.title(\"Occupation and Purchase Analysis\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Although there are some occupations which have higher representations, it seems that the amount each user spends on average is more or less the same for all occupations. Of course, in the end, occupations with the highest representations will have the highest amounts of purchases."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### City_Category"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 147,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(data['City_Category'])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "It is observed that city category B has made the most number of puchases."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 148,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "data.groupby(\"City_Category\").mean()[\"Purchase\"].plot(kind='bar')\n",
    "plt.title(\"City Category and Purchase Analysis\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "However, the city whose buyers spend the most is city type ‘C’."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Stay_In_Current_City_Years"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 149,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(data['Stay_In_Current_City_Years'])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "It looks like the longest someone is living in that city the less prone they are to buy new things. Hence, if someone is new in town and needs a great number of new things for their house that they’ll take advantage of the low prices in Black Friday to purchase all the things needed."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 151,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "data.groupby(\"Stay_In_Current_City_Years\").mean()[\"Purchase\"].plot(kind='bar')\n",
    "plt.title(\"Stay_In_Current_City_Years and Purchase Analysis\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We see the same pattern seen before which show that on average people tend to spend the same amount on purchases regardeless of their group. People who are new in city are responsible for the higher number of purchase, however looking at it individually they tend to spend the same amount independently of how many years the have lived in their current city."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Age"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 152,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAZcAAAEWCAYAAACqitpwAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjMsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+AADFEAAAfyElEQVR4nO3debgcVZ3/8feHHYGwmAuGBIhoRok4ImQQBCSKDyQoAgoKPyFh8QcygDrgjOAGggjjggoKDCMxCSiIIBiULbKrbGFLAhGJGCEmQCAIYRENfOePcy4pOt2dTnK6O/fyeT1PPbf6VNWpb1VX+ps6VXVKEYGZmVlJK3U7ADMz63+cXMzMrDgnFzMzK87JxczMinNyMTOz4pxczMysOCcXe92QdI6krxSqa1NJz0laOX++UdKnStSd67tK0thS9S3Fer8u6UlJj3V63da/OLlYvyBplqQXJS2Q9DdJv5f0aUmvHuMR8emIOLnFuj7YbJ6IeCQi1o6IlwvEfqKkC2rqHx0RE5a37qWMYxPgWGB4RLypyXxvlvSKpLM6F531NU4u1p/sERHrAJsBpwFfAM4rvRJJq5SucwWxGfBURDyxhPnGAE8D+0lavf1hWV/k5GL9TkQ8ExGTgE8AYyVtCSBpvKSv5/GBkn6Vz3LmS7pF0kqSzgc2Ba7IzV7/JWmopJB0qKRHgOsrZdVE8xZJd0h6RtIvJW2Q1zVS0uxqjL1nR5JGAV8EPpHXd1+e/mozW47ry5L+IukJSRMlrZun9cYxVtIjuUnrS432jaR18/Lzcn1fzvV/EJgMbJzjGN9kF48Bvgz8E9ijpv5dJT2Y98FZkm6qNhdKOkTSDElPS7pG0mZN1mN9mJOL9VsRcQcwG9ipzuRj87QeYCPSD3xExIHAI6SzoLUj4puVZXYGtgB2a7DKMcAhwMbAQuCMFmK8GvgG8LO8vnfVme2gPLwf2BxYG/hBzTw7Am8DdgG+KmmLBqs8E1g317NzjvngiPgNMBqYk+M4qN7CknYChgAXARfn5XunDQQuAY4H3gg8CLy3Mn0v0n7+KGm/3wJc2CBO6+OcXKy/mwNsUKf8n8AgYLOI+GdE3BJL7mjvxIh4PiJebDD9/IiYHhHPA18BPt57wX85fRI4PSIejojnSD/e+9WcNX0tIl6MiPuA+4DFklSO5RPA8RGxICJmAd8BDlyKWMYCV0XE08BPgdGSNszTdgfuj4hfRERvcq3eGHA4cGpEzMjTvwFs5bOX/snJxfq7wcD8OuXfAmYC10p6WNJxLdT16FJM/wuwKjCwpSib2zjXV617FdIZV6/qj/gLpLObWgOB1erUNbiVICStCewL/AQgIm4lneX9v0qcr+6DnKyrzYGbAd/PTZF/I30vanX91rc4uVi/JenfSD9cv62dlv/nfmxEbE66bnCMpF16JzeocklnNptUxjclnR09CTwPvKES18qkZqFW651D+mGu1r0QeHwJy9V6MsdUW9dfW1x+b2AAcJakx/LtyoNZ1DQ2l9RkBoAkVT+TEs/hEbFeZVgzIn6/lNthfYCTi/U7kgZI+jDpusAFETGtzjwflvTW/AP4LPByHiD9aG++DKs+QNJwSW8ATgIuybcq/xFYQ9KHJK1KuhhevcvqcWBo9bbpGhcC/5FvAV6bRddoFi5NcDmWi4FTJK2Tm6OOAS5ovuSrxgLjgHcCW+VhB1LT1juBXwPvlLRXbrI7Eqje0nwOcLykd8CrNxfsuzTbYH2Hk4v1J1dIWkD6H/KXgNOBgxvMOwz4DfAccCtwVkTcmKedCnw5N998finWfz4wntREtQbwGUh3rwH/DvyIdJbwPK9tLvp5/vuUpLvr1Dsu130z8Gfg78DRSxFX1dF5/Q+Tzuh+mutvStJg0s0C34uIxyrDXcDVwNiIeJLUbPZN4ClgODAFeAkgIi4D/hu4SNKzwHTSTQTWD8kvCzOzdshnYrOBT0bEDd2OxzrLZy5mVoyk3SStlx+u/CLpgv1tXQ7LusDJxcxK2h74E+nmgT2AvZrcum39mJvFzMysOJ+5mJlZcf21A76lNnDgwBg6dGi3wzAz61PuuuuuJyOip7bcySUbOnQoU6ZM6XYYZmZ9iqS/1Ct3s5iZmRXn5GJmZsU5uZiZWXFOLmZmVpyTi5mZFefkYmZmxTm5mJlZcU4uZmZWnJOLmZkV5yf0bYW0w5k7dDuEhn539O+6HYLZCs9nLmZmVpyTi5mZFefkYmZmxTm5mJlZcU4uZmZWnJOLmZkV5+RiZmbFObmYmVlxTi5mZlack4uZmRXXtuQiaRNJN0iaIel+SZ/N5RtImizpofx3/VwuSWdImilpqqStK3WNzfM/JGlspXwbSdPyMmdIUrN1mJlZZ7TzzGUhcGxEbAFsBxwpaThwHHBdRAwDrsufAUYDw/JwGHA2pEQBnAC8B9gWOKGSLM7O8/YuNyqXN1qHmZl1QNuSS0TMjYi78/gCYAYwGNgTmJBnmwDslcf3BCZGchuwnqRBwG7A5IiYHxFPA5OBUXnagIi4NSICmFhTV711mJlZB3TkmoukocC7gduBjSJiLqQEBGyYZxsMPFpZbHYua1Y+u045TdZRG9dhkqZImjJv3rxl3TwzM6vR9uQiaW3gUuBzEfFss1nrlMUylLcsIs6NiBERMaKnp2dpFjUzsybamlwkrUpKLD+JiF/k4sdzkxb57xO5fDawSWXxIcCcJZQPqVPebB1mZtYB7bxbTMB5wIyIOL0yaRLQe8fXWOCXlfIx+a6x7YBncpPWNcCuktbPF/J3Ba7J0xZI2i6va0xNXfXWYWZmHdDON1HuABwITJN0by77InAacLGkQ4FHgH3ztCuB3YGZwAvAwQARMV/SycCdeb6TImJ+Hj8CGA+sCVyVB5qsw8zMOqBtySUifkv96yIAu9SZP4AjG9Q1DhhXp3wKsGWd8qfqrcPMzDrDT+ibmVlxTi5mZlack4uZmRXn5GJmZsU5uZiZWXFOLmZmVpyTi5mZFefkYmZmxTm5mJlZcU4uZmZWnJOLmZkV5+RiZmbFObmYmVlxTi5mZlack4uZmRXn5GJmZsU5uZiZWXFOLmZmVpyTi5mZFefkYmZmxTm5mJlZcU4uZmZWnJOLmZkV5+RiZmbFObmYmVlxTi5mZlack4uZmRXn5GJmZsU5uZiZWXFOLmZmVpyTi5mZFefkYmZmxTm5mJlZcU4uZmZWnJOLmZkV5+RiZmbFObmYmVlxTi5mZlack4uZmRXXtuQiaZykJyRNr5SdKOmvku7Nw+6VacdLminpQUm7VcpH5bKZko6rlL9Z0u2SHpL0M0mr5fLV8+eZefrQdm2jmZnV184zl/HAqDrl342IrfJwJYCk4cB+wDvyMmdJWlnSysAPgdHAcGD/PC/Af+e6hgFPA4fm8kOBpyPircB383xmZtZBbUsuEXEzML/F2fcELoqIlyLiz8BMYNs8zIyIhyPiH8BFwJ6SBHwAuCQvPwHYq1LXhDx+CbBLnt/MzDqkG9dcjpI0NTebrZ/LBgOPVuaZncsalb8R+FtELKwpf01defozef7FSDpM0hRJU+bNm7f8W2ZmZkDnk8vZwFuArYC5wHdyeb0zi1iG8mZ1LV4YcW5EjIiIET09Pc3iNjOzpdDR5BIRj0fEyxHxCvC/pGYvSGcem1RmHQLMaVL+JLCepFVqyl9TV56+Lq03z5mZWQEdTS6SBlU+7g303kk2Cdgv3+n1ZmAYcAdwJzAs3xm2Gumi/6SICOAGYJ+8/Fjgl5W6xubxfYDr8/xmZtYhqyx5lmUj6UJgJDBQ0mzgBGCkpK1IzVSzgMMBIuJ+SRcDDwALgSMj4uVcz1HANcDKwLiIuD+v4gvARZK+DtwDnJfLzwPOlzSTdMayX7u20czM6mtbcomI/esUn1enrHf+U4BT6pRfCVxZp/xhFjWrVcv/Duy7VMGamVlRfkLfzMyKc3IxM7PinFzMzKw4JxczMyvOycXMzIpzcjEzs+KcXMzMrDgnFzMzK87JxczMinNyMTOz4pxczMysOCcXMzMrzsnFzMyKc3IxM7PinFzMzKw4JxczMyvOycXMzIpzcjEzs+KcXMzMrLiWkouk61opMzMzA1il2URJawBvAAZKWh9QnjQA2LjNsZmZWR/VNLkAhwOfIyWSu1iUXJ4FftjGuMzMrA9rmlwi4vvA9yUdHRFndigmMzPr45Z05gJARJwp6b3A0OoyETGxTXGZmVkf1lJykXQ+8BbgXuDlXByAk4uZmS2mpeQCjACGR0S0MxgzM+sfWn3OZTrwpnYGYmZm/UerZy4DgQck3QG81FsYER9pS1RmZtantZpcTmxnEGZm1r+0erfYTe0OxMzM+o9W7xZbQLo7DGA1YFXg+YgY0K7AzMys72r1zGWd6mdJewHbtiUiMzPr85apV+SIuBz4QOFYzMysn2i1WeyjlY8rkZ578TMvZmZWV6t3i+1RGV8IzAL2LB6NmZn1C61eczm43YGYmZVyygH7dDuEur50wSXdDqFjWn1Z2BBJl0l6QtLjki6VNKTdwZmZWd/U6gX9HwOTSO91GQxckcvMzMwW02py6YmIH0fEwjyMB3raGJeZmfVhrSaXJyUdIGnlPBwAPNVsAUnjcjPa9ErZBpImS3oo/10/l0vSGZJmSpoqaevKMmPz/A9JGlsp30bStLzMGZLUbB1mZtY5rSaXQ4CPA48Bc4F9gCVd5B8PjKopOw64LiKGAdflzwCjgWF5OAw4G1KiAE4A3kN6aPOESrI4O8/bu9yoJazDzMw6pNXkcjIwNiJ6ImJDUrI5sdkCEXEzML+meE9gQh6fAOxVKZ8YyW3AepIGAbsBkyNifkQ8DUwGRuVpAyLi1vyOmYk1ddVbh5mZdUiryeVf8487ABExH3j3Mqxvo4iYm+uYC2yYywcDj1bmm53LmpXPrlPebB2LkXSYpCmSpsybN28ZNsfMzOppNbmsVL12kZurWn0AsxWqUxbLUL5UIuLciBgRESN6enx/gplZKa0miO8Av5d0CelH/OPAKcuwvsclDYqIublp64lcPhvYpDLfEGBOLh9ZU35jLh9SZ/5m6zAzsw5p9Qn9iZKmkDqrFPDRiHhgGdY3CRgLnJb//rJSfpSki0gX75/JyeEa4BuVs6ZdgeMjYr6kBZK2A24HxgBnLmEdZmZ9woxTru92CHVt8aXW+ytuuWkrJ5OWE4qkC0lnHQMlzSbd9XUacLGkQ4FHgH3z7FcCuwMzgRfId6LlJHIycGee76R8vQfgCNIdaWsCV+WBJuswM7MOKXnd5DUiYv8Gk3apM28ARzaoZxwwrk75FGDLOuVP1VuHmZl1zjK9z8XMzKwZJxczMyvOycXMzIpzcjEzs+KcXMzMrDgnFzMzK87JxczMinNyMTOz4pxczMysOCcXMzMrzsnFzMyKc3IxM7PinFzMzKw4JxczMyvOycXMzIpzcjEzs+KcXMzMrDgnFzMzK87JxczMinNyMTOz4pxczMysOCcXMzMrzsnFzMyKc3IxM7PinFzMzKw4JxczMyvOycXMzIpzcjEzs+KcXMzMrDgnFzMzK87JxczMinNyMTOz4pxczMysOCcXMzMrzsnFzMyKc3IxM7PinFzMzKw4JxczMyvOycXMzIrrSnKRNEvSNEn3SpqSyzaQNFnSQ/nv+rlcks6QNFPSVElbV+oZm+d/SNLYSvk2uf6ZeVl1fivNzF6/unnm8v6I2CoiRuTPxwHXRcQw4Lr8GWA0MCwPhwFnQ0pGwAnAe4BtgRN6E1Ke57DKcqPavzlmZtZrRWoW2xOYkMcnAHtVyidGchuwnqRBwG7A5IiYHxFPA5OBUXnagIi4NSICmFipy8zMOqBbySWAayXdJemwXLZRRMwFyH83zOWDgUcry87OZc3KZ9cpNzOzDlmlS+vdISLmSNoQmCzpD03mrXe9JJahfPGKU2I7DGDTTTdtHrGZmbWsK2cuETEn/30CuIx0zeTx3KRF/vtEnn02sEll8SHAnCWUD6lTXi+OcyNiRESM6OnpWd7NMjOzrOPJRdJaktbpHQd2BaYDk4DeO77GAr/M45OAMfmuse2AZ3Kz2TXArpLWzxfydwWuydMWSNou3yU2plKXmZl1QDeaxTYCLst3B68C/DQirpZ0J3CxpEOBR4B98/xXArsDM4EXgIMBImK+pJOBO/N8J0XE/Dx+BDAeWBO4Kg9mZtYhHU8uEfEw8K465U8Bu9QpD+DIBnWNA8bVKZ8CbLncwZqZ2TLp1gV9s37tpvft3O0Q6tr55pu6HYK9TqxIz7mYmVk/4eRiZmbFObmYmVlxTi5mZlack4uZmRXn5GJmZsX5VmQzW8wPjr2i2yHUddR39uh2CNYin7mYmVlxTi5mZlack4uZmRXn5GJmZsU5uZiZWXFOLmZmVpyTi5mZFefkYmZmxTm5mJlZcU4uZmZWnJOLmZkV5+RiZmbFObmYmVlxTi5mZlack4uZmRXn97n0Y4+c9M5uh1DXpl+d1u0QzKzNfOZiZmbFObmYmVlxTi5mZlack4uZmRXn5GJmZsX5brEmtvnPid0Ooa67vjWm2yGYmTXlMxczMyvOycXMzIpzcjEzs+KcXMzMrDgnFzMzK87JxczMinNyMTOz4pxczMysOCcXMzMrzsnFzMyK67fJRdIoSQ9KminpuG7HY2b2etIvk4uklYEfAqOB4cD+koZ3Nyozs9ePfplcgG2BmRHxcET8A7gI2LPLMZmZvW4oIrodQ3GS9gFGRcSn8ucDgfdExFE18x0GHJY/vg14sI1hDQSebGP97eb4u6cvxw6Ov9vaHf9mEdFTW9hfu9xXnbLFsmhEnAuc2/5wQNKUiBjRiXW1g+Pvnr4cOzj+butW/P21WWw2sEnl8xBgTpdiMTN73emvyeVOYJikN0taDdgPmNTlmMzMXjf6ZbNYRCyUdBRwDbAyMC4i7u9yWB1pfmsjx989fTl2cPzd1pX4++UFfTMz667+2ixmZmZd5ORiZmbFObksp1a6mZF0taS/SfpVTfktku7NwxxJl3cm6sXimyVpWo5jSi47UdJfK/Ht3o3YaknaRNINkmZIul/SZyvTjs7fxf2Svtlg+ZMlTc3bdK2kjXP5SEnPVLb3q23chpUl3dN7PCg5RdIf83Z9psFy4yX9uRLjVpXlz8jH4FRJW7cx9nrHyr55n78iqeEtr42OKUlDJb1YKT+nTbGvIekOSffleL+Wy1va/5V6zpT0XOXzQZLmVeL/VJviHyfpCUnTK2VbSbqt9/uQtG2DZX+S/21Mz/Wsmsvbd9xHhIdlHEg3C/wJ2BxYDbgPGF5nvl2APYBfNanrUmBMl7ZjFjCwpuxE4PNLWO5E4KAOxzoI2DqPrwP8kdTFz/uB3wCr52kbNlh+QGX8M8A5eXxks++n8DYcA/y0d33AwcBEYKUlxD4e2KdO+e7AVaTnu7YDbu/wsbIF6SHkG4ERSzheFjumgKHA9A7sdwFr5/FVgdvz/mpp/+dpI4DzgecqZQcBP+hA/O8Dtq7uK+BaYHTlOLixwbK75+0XcCFwRC5v23HvM5fl01I3MxFxHbCgUSWS1gE+AHTlzKUviYi5EXF3Hl8AzAAGA0cAp0XES3naEw2Wf7bycS3qPFzbTpKGAB8CflQpPgI4KSJegcaxN7EnMDGS24D1JA0qEnALImJGRLSzd4si8v7pPeNYNQ9Bi/tfqc/CbwH/1YFwFxMRNwPza4uBAXl8XRo8zxcRV+btD+AO0rN/beXksnwGA49WPs/OZUtrb+C6mh++TgrgWkl3KXWJ0+uo3MwyTtL6XYqtIUlDgXeT/gf6L8BOkm6XdJOkf2uy3CmSHgU+CVSbAbbPTSZXSXpHm8L+HunH6ZVK2VuAT+RmjaskDWuy/Cn5O/mupNVzWanjsBWNjpVWNTqm3pybCm+StFOhWBeTmyTvBZ4AJkfE7bS+/48CJkXE3DrTPpa36xJJm9SZ3i6fA76Vj+dvA8c3mzk3hx0IXF0pbs9x3+5Tuf48APsCP6p8PhA4s8G8I2lw+klq0vhYF7dj4/x3Q1LT3vuAjUjNfisBp5CeFQJ4J3BvHh4DHql8fmMHY14buAv4aP48HTiDdNq/LfBn8q32Teo4HvhaHh/AoiaT3YGH2hDzh4Gzao8H4Dng2Dz+UeCWBssPytu3OjAB+Gou/zWwY2W+64BtOnWsVKbdSPNmsUbH1Oq9xw6wDSlRDmhH/JVY1gNuALZsZf8DGwO/BVbp/c4q097IoubYTwPXtzHuoby2WeyM3t8O4OPAb5aw/P8C36t8bttx37Yv7/UwANsD11Q+Hw+cUPmx/Uhl2qs/JjV1vBF4Clij29uT4zmRmnbx2gO6Zt6DuhDjqqQHZI+plF0NjKx8/hPQA/w4fxdX1qlns3rblafNoubaQoG4TyWdVcwiJeYXgAuAPwBD8zwCnsnj1+TYf1SnrlePJ+B/gP0r0x4EBnX6WKEmuSxh39c9purV08b4TwA+38r+JzVlPpa/u1mkM8+ZdepcuXf5NsX8mv0GPMOi5xUFPNvo2Mnbezn52lKD+osd924WWz71upm5JCK2ykMrXc7sS/qR+HtbI21A0lr5mg+S1gJ2BabXtNnvTToz6DpJAs4DZkTE6ZVJl5OuWyHpX0g3WDwZEQfn76L3zqRqk8dHSD8sSHpTrpt8x81KpKRfTEQcHxFDImIo6Vi5PiIOqMYO7Ey6SYGI2C3H3tu796DKPtiLRd/JJGBMvutpO9KPW72mm+XS6FhpNH+dfV/3mJLUk69nIGlzYBjwcBvi75G0Xh5fE/gg6ftf4v6PiF9HxJsiYmj+/l6IiLfW2a6PkK4DdsqcHDOkbXioNvYc46eA3Uj/CXm1Sbatx327/3fQ3wfSqeQfSf9T/lKDeW4B5gEvkv7nultl2o2k1wN0K/7NSc0b9wH3924D6Y6YacBU0o/XYv8Tpjt3i+1IavefyqIzxN1JyeQC0g/W3cAHGix/aZ5nKnAFMDiXH5W3/z7gNuC9bd6OkSw681iP1LQ1DbgVeFeDZa7P80zP29rbnCHSy/H+lKe35X/9TY6VvfNx/RLwOJWz+Zrl6x5TwMcq+/5uYI82xf+vwD15/dNZ1KzY0v6vqavaLHZqJf4bgLe3Kf4LgbnAP/P+PjT/e7grr/t2GjSHAgvz8dH7b6Z329t23Lv7FzMzK87NYmZmVpyTi5mZFefkYmZmxTm5mJlZcU4uZmZWnJOLrZAkvZx7ab0/d01xjKSV8rQRks7I46tL+k2e9xOSdsrL3JufZWhXfCMlvbfJ9NG5O5EZkv4g6dvLU19Jkq7sfd6jDXX35C547qnXjUue/k9Jh7dj/bbi6JevObZ+4cWI6O1SfkNSL8LrAidExBRgSp7v3cCqlXnPAb4dET9uZSX5ATJF5cGyFo0kdRvy+zp1bgn8APhQRPxB0irAkvrhalhfKZVtbefrE3YB/hARYxtM35f0PMX+pJ4FrL9qx8M+Hjws70DlIbX8eXPSk8MiP4BI6t9qJqkLjHuBw0m9xv4Z+Ele7j9JPSlMZVE/YkNJT1GfRXqobjPS0+a3kh7i+zmLHlCcBXwtl08D3p6Xfwz4a17vTjWxTgQOabBde5AedruH9IqAjerVR+q65tIc+53ADnn5HmByjud/gL+Qu+sgdeU/PQ+fa7KtsyrLHEDqJffeXN/KeRif65kG/Eed7diM1IfZ1Px3U2ArUl9z83J9a9ZZ7hZyb+LkB1hz+aGkh5FvJPV/9YPK9i62Hzys+EPXA/Dgod5ATXLJZU/nH+ORLHq6/dXx/Hk8+Z0nOWGcS0pIK5ES0vvyD+4rwHZ5voHAzcBa+fMXWPQE8yzg6Dz+7+S+mmjyvpv8w9/oKfv1WdQX1KeA79Srj3SmtmMe35TU3Q2kM6Lj8/goUm8FA0kdPk4jvUZgbdJT1++u3dbKNg0kvYflCtKZH6QENCbXNbky/3p1tuMKYGwePwS4PI8fRIN3mwCbkDtGBL5B7huO1CnkLGADUr9xt1SSS9394GHFH9wsZn2JlnL+XfNwT/68NqnfqkeAv0R69wmkF0YNB36Xu1lajXQW0+sX+e9dpF5zl8cQ4Ge5P6rVSGdZ9XwQGJ7jARiQ+/XakdTdChFxtaSn8/Qdgcsi4nkASb8gnQFN4rXbWrULKZHcmdezJqkr+iuAzSWdSeoW5do6y27Pon1xPlD3zZ819gMuzuMXkfqIO510JnNTRMzPsf+c9AqFhvsh0rt8bAXm5GJ9Qu7Q8GXSj98WrS4GnBoRr2nbV3oPzPM1802OiP0b1PNS/vsyrf2buZ/0o31fnWlnAqdHxCRJI0lnLPWsBGwfES/WxN4owTZLvM83KBcwISIWeweIpHeROjo8ktSV+yFN6ofWXrq2P7CRpE/mzxvnjkSbxV53P9iKz3eL2QpPUg9wDqmpZGk6w7sGOETS2rmewfnmgFq3ATtI6u3l9g25Z+VmFpBes1zPt4Av9tYhaSVJx+Rp65KurQBUL3rX1nctqVNBch1b5dHfkn7skbQrqZkNUrPeXjn2tUhnN7csYRuuA/bp3SeSNpC0maSBpG7ZLwW+Qnq1bq3fk85EIL107bfNViTpbaRmx8GxqGfhU3MddwA7S1o/3/zwsRb2g63gnFxsRbVm763IpAvf15IurLcsIq4ltdnfKmkacAl1EkJEzCNdK7hQ0lRSsnn7Eqq/Atg7x/iaW24jYirpDYEXSppBujDe2y37icDPJd0CPNmkvs8AI5TebvgA6SVUkPbBrpLuBkaTesldEOnVz+NJP9S3k64N3UMTEfEA8GXSmyWnkm4UGER6i+WNSm9sHE/9txt+Bjg4L3cg8Nlm6yKdtVxWU3YpqQv4v5KuwdxO+q4fIN2k0bueevvBVnDuFdmsD1F6tfHLEbFQ0vbA2ZFvw+7LJK0dEc/lM5fLSG+prE1G1of4motZ37IpcHF+oPQfwP/vcjylnCjpg8AapLPUy7scjy0nn7mYmVlxvuZiZmbFObmYmVlxTi5mZlack4uZmRXn5GJmZsX9HxYkKkzr84thAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.countplot(data['Age'])\n",
    "plt.title('Distribution of Age')\n",
    "plt.xlabel('Different Categories of Age')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Age 26-35 Age group makes the most no of purchases in the age group."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 153,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x19d8759c148>"
      ]
     },
     "execution_count": 153,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "data.groupby(\"Age\").mean()[\"Purchase\"].plot(kind='bar')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Mean puchase rate between the age groups tends to be the same except that the 51-55 age group has a little higher average purchase amount"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 154,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "data.groupby(\"Age\").sum()['Purchase'].plot(kind=\"bar\")\n",
    "plt.title(\"Age and Purchase Analysis\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Total amount spent in purchase is in accordance with the number of purchases made, distributed by age."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Product_Category_1"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 155,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1296x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(18,5))\n",
    "sns.countplot(data['Product_Category_1'])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "It is clear that `Product_Category_1` numbers 1,5 and 8 stand out. Unfortunately we don't know which product each number represents as it is masked."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 156,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1296x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "data.groupby('Product_Category_1').mean()['Purchase'].plot(kind='bar',figsize=(18,5))\n",
    "plt.title(\"Product_Category_1 and Purchase Mean Analysis\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "If you see the value spent on average for Product_Category_1 you see that although there were more products bought for categories 1,5,8 the average amount spent for those three is not the highest. It is interesting to see other categories appearing with high purchase values despite having low impact on sales number."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 157,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1296x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "data.groupby('Product_Category_1').sum()['Purchase'].plot(kind='bar',figsize=(18,5))\n",
    "plt.title(\"Product_Category_1 and Purchase Analysis\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The distribution that we saw for this predictor previously appears here. For example, those three products have the highest sum of sales since their were three most sold products."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Product_Category_2"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 158,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1296x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(18,5))\n",
    "sns.countplot(data['Product_Category_2'])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Product_Category_3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 159,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 1296x360 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "plt.figure(figsize=(18,5))\n",
    "sns.countplot(data['Product_Category_3'])\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 160,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>User_ID</th>\n",
       "      <th>Occupation</th>\n",
       "      <th>Marital_Status</th>\n",
       "      <th>Product_Category_1</th>\n",
       "      <th>Product_Category_2</th>\n",
       "      <th>Product_Category_3</th>\n",
       "      <th>Purchase</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>User_ID</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.023024</td>\n",
       "      <td>0.018732</td>\n",
       "      <td>0.003687</td>\n",
       "      <td>0.001471</td>\n",
       "      <td>0.004045</td>\n",
       "      <td>0.005389</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>Occupation</td>\n",
       "      <td>-0.023024</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.024691</td>\n",
       "      <td>-0.008114</td>\n",
       "      <td>-0.000031</td>\n",
       "      <td>0.013452</td>\n",
       "      <td>0.021104</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>Marital_Status</td>\n",
       "      <td>0.018732</td>\n",
       "      <td>0.024691</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.020546</td>\n",
       "      <td>0.015116</td>\n",
       "      <td>0.019452</td>\n",
       "      <td>0.000129</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>Product_Category_1</td>\n",
       "      <td>0.003687</td>\n",
       "      <td>-0.008114</td>\n",
       "      <td>0.020546</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.540423</td>\n",
       "      <td>0.229490</td>\n",
       "      <td>-0.314125</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>Product_Category_2</td>\n",
       "      <td>0.001471</td>\n",
       "      <td>-0.000031</td>\n",
       "      <td>0.015116</td>\n",
       "      <td>0.540423</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>0.543544</td>\n",
       "      <td>-0.209973</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>Product_Category_3</td>\n",
       "      <td>0.004045</td>\n",
       "      <td>0.013452</td>\n",
       "      <td>0.019452</td>\n",
       "      <td>0.229490</td>\n",
       "      <td>0.543544</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>-0.022257</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>Purchase</td>\n",
       "      <td>0.005389</td>\n",
       "      <td>0.021104</td>\n",
       "      <td>0.000129</td>\n",
       "      <td>-0.314125</td>\n",
       "      <td>-0.209973</td>\n",
       "      <td>-0.022257</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                     User_ID  Occupation  Marital_Status  Product_Category_1  \\\n",
       "User_ID             1.000000   -0.023024        0.018732            0.003687   \n",
       "Occupation         -0.023024    1.000000        0.024691           -0.008114   \n",
       "Marital_Status      0.018732    0.024691        1.000000            0.020546   \n",
       "Product_Category_1  0.003687   -0.008114        0.020546            1.000000   \n",
       "Product_Category_2  0.001471   -0.000031        0.015116            0.540423   \n",
       "Product_Category_3  0.004045    0.013452        0.019452            0.229490   \n",
       "Purchase            0.005389    0.021104        0.000129           -0.314125   \n",
       "\n",
       "                    Product_Category_2  Product_Category_3  Purchase  \n",
       "User_ID                       0.001471            0.004045  0.005389  \n",
       "Occupation                   -0.000031            0.013452  0.021104  \n",
       "Marital_Status                0.015116            0.019452  0.000129  \n",
       "Product_Category_1            0.540423            0.229490 -0.314125  \n",
       "Product_Category_2            1.000000            0.543544 -0.209973  \n",
       "Product_Category_3            0.543544            1.000000 -0.022257  \n",
       "Purchase                     -0.209973           -0.022257  1.000000  "
      ]
     },
     "execution_count": 160,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.corr()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## HeatMap"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 161,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.heatmap(data.corr(),annot=True)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "There is a some corellation between the product category groups."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 162,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['User_ID', 'Product_ID', 'Gender', 'Age', 'Occupation', 'City_Category',\n",
       "       'Stay_In_Current_City_Years', 'Marital_Status', 'Product_Category_1',\n",
       "       'Product_Category_2', 'Product_Category_3', 'Purchase'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 162,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 163,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = data.copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 164,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>User_ID</th>\n",
       "      <th>Product_ID</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Occupation</th>\n",
       "      <th>City_Category</th>\n",
       "      <th>Stay_In_Current_City_Years</th>\n",
       "      <th>Marital_Status</th>\n",
       "      <th>Product_Category_1</th>\n",
       "      <th>Product_Category_2</th>\n",
       "      <th>Product_Category_3</th>\n",
       "      <th>Purchase</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>1000001</td>\n",
       "      <td>P00069042</td>\n",
       "      <td>F</td>\n",
       "      <td>0-17</td>\n",
       "      <td>10</td>\n",
       "      <td>A</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>8370</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>1000001</td>\n",
       "      <td>P00248942</td>\n",
       "      <td>F</td>\n",
       "      <td>0-17</td>\n",
       "      <td>10</td>\n",
       "      <td>A</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>6.0</td>\n",
       "      <td>14.0</td>\n",
       "      <td>15200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>1000001</td>\n",
       "      <td>P00087842</td>\n",
       "      <td>F</td>\n",
       "      <td>0-17</td>\n",
       "      <td>10</td>\n",
       "      <td>A</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>12</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1422</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>1000001</td>\n",
       "      <td>P00085442</td>\n",
       "      <td>F</td>\n",
       "      <td>0-17</td>\n",
       "      <td>10</td>\n",
       "      <td>A</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>12</td>\n",
       "      <td>14.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1057</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>1000002</td>\n",
       "      <td>P00285442</td>\n",
       "      <td>M</td>\n",
       "      <td>55+</td>\n",
       "      <td>16</td>\n",
       "      <td>C</td>\n",
       "      <td>4+</td>\n",
       "      <td>0</td>\n",
       "      <td>8</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>7969</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   User_ID Product_ID Gender   Age  Occupation City_Category  \\\n",
       "0  1000001  P00069042      F  0-17          10             A   \n",
       "1  1000001  P00248942      F  0-17          10             A   \n",
       "2  1000001  P00087842      F  0-17          10             A   \n",
       "3  1000001  P00085442      F  0-17          10             A   \n",
       "4  1000002  P00285442      M   55+          16             C   \n",
       "\n",
       "  Stay_In_Current_City_Years  Marital_Status  Product_Category_1  \\\n",
       "0                          2               0                   3   \n",
       "1                          2               0                   1   \n",
       "2                          2               0                  12   \n",
       "3                          2               0                  12   \n",
       "4                         4+               0                   8   \n",
       "\n",
       "   Product_Category_2  Product_Category_3  Purchase  \n",
       "0                 NaN                 NaN      8370  \n",
       "1                 6.0                14.0     15200  \n",
       "2                 NaN                 NaN      1422  \n",
       "3                14.0                 NaN      1057  \n",
       "4                 NaN                 NaN      7969  "
      ]
     },
     "execution_count": 164,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 165,
   "metadata": {},
   "outputs": [],
   "source": [
    "# df['Stay_In_Current_City_Years'] = df['Stay_In_Current_City_Years'].replace(to_replace=\"4+\",value=\"4\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 166,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Dummy Variables:\n",
    "df = pd.get_dummies(df, columns=['Stay_In_Current_City_Years'])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Encoding the categorical variables"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 167,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import LabelEncoder\n",
    "lr = LabelEncoder()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 168,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['Gender'] = lr.fit_transform(df['Gender'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 169,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['Age'] = lr.fit_transform(df['Age'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 170,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['City_Category'] = lr.fit_transform(df['City_Category'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 171,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>User_ID</th>\n",
       "      <th>Product_ID</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>Occupation</th>\n",
       "      <th>City_Category</th>\n",
       "      <th>Marital_Status</th>\n",
       "      <th>Product_Category_1</th>\n",
       "      <th>Product_Category_2</th>\n",
       "      <th>Product_Category_3</th>\n",
       "      <th>Purchase</th>\n",
       "      <th>Stay_In_Current_City_Years_0</th>\n",
       "      <th>Stay_In_Current_City_Years_1</th>\n",
       "      <th>Stay_In_Current_City_Years_2</th>\n",
       "      <th>Stay_In_Current_City_Years_3</th>\n",
       "      <th>Stay_In_Current_City_Years_4+</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>1000001</td>\n",
       "      <td>P00069042</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>10</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>3</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>8370</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>1000001</td>\n",
       "      <td>P00248942</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>10</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>6.0</td>\n",
       "      <td>14.0</td>\n",
       "      <td>15200</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>1000001</td>\n",
       "      <td>P00087842</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>10</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>12</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1422</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>1000001</td>\n",
       "      <td>P00085442</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>10</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>12</td>\n",
       "      <td>14.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>1057</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>1000002</td>\n",
       "      <td>P00285442</td>\n",
       "      <td>1</td>\n",
       "      <td>6</td>\n",
       "      <td>16</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>8</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>7969</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   User_ID Product_ID  Gender  Age  Occupation  City_Category  Marital_Status  \\\n",
       "0  1000001  P00069042       0    0          10              0               0   \n",
       "1  1000001  P00248942       0    0          10              0               0   \n",
       "2  1000001  P00087842       0    0          10              0               0   \n",
       "3  1000001  P00085442       0    0          10              0               0   \n",
       "4  1000002  P00285442       1    6          16              2               0   \n",
       "\n",
       "   Product_Category_1  Product_Category_2  Product_Category_3  Purchase  \\\n",
       "0                   3                 NaN                 NaN      8370   \n",
       "1                   1                 6.0                14.0     15200   \n",
       "2                  12                 NaN                 NaN      1422   \n",
       "3                  12                14.0                 NaN      1057   \n",
       "4                   8                 NaN                 NaN      7969   \n",
       "\n",
       "   Stay_In_Current_City_Years_0  Stay_In_Current_City_Years_1  \\\n",
       "0                             0                             0   \n",
       "1                             0                             0   \n",
       "2                             0                             0   \n",
       "3                             0                             0   \n",
       "4                             0                             0   \n",
       "\n",
       "   Stay_In_Current_City_Years_2  Stay_In_Current_City_Years_3  \\\n",
       "0                             1                             0   \n",
       "1                             1                             0   \n",
       "2                             1                             0   \n",
       "3                             1                             0   \n",
       "4                             0                             0   \n",
       "\n",
       "   Stay_In_Current_City_Years_4+  \n",
       "0                              0  \n",
       "1                              0  \n",
       "2                              0  \n",
       "3                              0  \n",
       "4                              1  "
      ]
     },
     "execution_count": 171,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 173,
   "metadata": {},
   "outputs": [],
   "source": [
    "df['Product_Category_2'] =df['Product_Category_2'].fillna(0).astype('int64')\n",
    "df['Product_Category_3'] =df['Product_Category_3'].fillna(0).astype('int64')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 174,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "User_ID                          0\n",
       "Product_ID                       0\n",
       "Gender                           0\n",
       "Age                              0\n",
       "Occupation                       0\n",
       "City_Category                    0\n",
       "Marital_Status                   0\n",
       "Product_Category_1               0\n",
       "Product_Category_2               0\n",
       "Product_Category_3               0\n",
       "Purchase                         0\n",
       "Stay_In_Current_City_Years_0     0\n",
       "Stay_In_Current_City_Years_1     0\n",
       "Stay_In_Current_City_Years_2     0\n",
       "Stay_In_Current_City_Years_3     0\n",
       "Stay_In_Current_City_Years_4+    0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 174,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 175,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 537577 entries, 0 to 537576\n",
      "Data columns (total 16 columns):\n",
      "User_ID                          537577 non-null int64\n",
      "Product_ID                       537577 non-null object\n",
      "Gender                           537577 non-null int32\n",
      "Age                              537577 non-null int32\n",
      "Occupation                       537577 non-null int64\n",
      "City_Category                    537577 non-null int32\n",
      "Marital_Status                   537577 non-null int64\n",
      "Product_Category_1               537577 non-null int64\n",
      "Product_Category_2               537577 non-null int64\n",
      "Product_Category_3               537577 non-null int64\n",
      "Purchase                         537577 non-null int64\n",
      "Stay_In_Current_City_Years_0     537577 non-null uint8\n",
      "Stay_In_Current_City_Years_1     537577 non-null uint8\n",
      "Stay_In_Current_City_Years_2     537577 non-null uint8\n",
      "Stay_In_Current_City_Years_3     537577 non-null uint8\n",
      "Stay_In_Current_City_Years_4+    537577 non-null uint8\n",
      "dtypes: int32(3), int64(7), object(1), uint8(5)\n",
      "memory usage: 41.5+ MB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Dropping the irrelevant columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 176,
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df.drop([\"User_ID\",\"Product_ID\"],axis=1)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Splitting data into independent and dependent variables"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 179,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df.drop(\"Purchase\",axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 180,
   "metadata": {},
   "outputs": [],
   "source": [
    "y=df['Purchase']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 181,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Modeling"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Linear Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 182,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LinearRegression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 183,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)"
      ]
     },
     "execution_count": 183,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "lr = LinearRegression()\n",
    "lr.fit(X_train,y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 184,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "9392.78408085134"
      ]
     },
     "execution_count": 184,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "lr.intercept_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 185,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([ 481.31865517,  107.64157841,    5.13000529,  336.95273272,\n",
       "        -63.3778221 , -317.00345883,    7.9238667 ,  148.12973485,\n",
       "        -32.78694504,   -1.66930455,   34.63808922,  -12.31969823,\n",
       "         12.13785861])"
      ]
     },
     "execution_count": 185,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "lr.coef_"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 186,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred = lr.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 187,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import mean_absolute_error,mean_squared_error, r2_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 188,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "3540.3993734221553"
      ]
     },
     "execution_count": 188,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mean_absolute_error(y_test, y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 189,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "21342855.359792948"
      ]
     },
     "execution_count": 189,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mean_squared_error(y_test, y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 190,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.13725207799200811"
      ]
     },
     "execution_count": 190,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "r2_score(y_test, y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 191,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "RMSE of Linear Regression Model is  4619.8328281219165\n"
     ]
    }
   ],
   "source": [
    "from math import sqrt\n",
    "print(\"RMSE of Linear Regression Model is \",sqrt(mean_squared_error(y_test, y_pred)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# DecisionTreeRegressor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 192,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.tree import DecisionTreeRegressor\n",
    "\n",
    "# create a regressor object \n",
    "regressor = DecisionTreeRegressor(random_state = 0)  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 193,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=None,\n",
       "                      max_leaf_nodes=None, min_impurity_decrease=0.0,\n",
       "                      min_impurity_split=None, min_samples_leaf=1,\n",
       "                      min_samples_split=2, min_weight_fraction_leaf=0.0,\n",
       "                      presort=False, random_state=0, splitter='best')"
      ]
     },
     "execution_count": 193,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "regressor.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 194,
   "metadata": {},
   "outputs": [],
   "source": [
    "dt_y_pred = regressor.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 195,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2403.1409470088884"
      ]
     },
     "execution_count": 195,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mean_absolute_error(y_test, dt_y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 196,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "11535194.335807195"
      ]
     },
     "execution_count": 196,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mean_squared_error(y_test, dt_y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 197,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.5337097695969879"
      ]
     },
     "execution_count": 197,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "r2_score(y_test, dt_y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 198,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "RMSE of Linear Regression Model is  3396.3501491759052\n"
     ]
    }
   ],
   "source": [
    "from math import sqrt\n",
    "print(\"RMSE of Linear Regression Model is \",sqrt(mean_squared_error(y_test, dt_y_pred)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Random Forest Regressor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 199,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.ensemble import RandomForestRegressor\n",
    "\n",
    "# create a regressor object \n",
    "RFregressor = RandomForestRegressor(random_state = 0)  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 200,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Nantha\\Anaconda3\\lib\\site-packages\\sklearn\\ensemble\\forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.\n",
      "  \"10 in version 0.20 to 100 in 0.22.\", FutureWarning)\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,\n",
       "                      max_features='auto', max_leaf_nodes=None,\n",
       "                      min_impurity_decrease=0.0, min_impurity_split=None,\n",
       "                      min_samples_leaf=1, min_samples_split=2,\n",
       "                      min_weight_fraction_leaf=0.0, n_estimators=10,\n",
       "                      n_jobs=None, oob_score=False, random_state=0, verbose=0,\n",
       "                      warm_start=False)"
      ]
     },
     "execution_count": 200,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "RFregressor.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 201,
   "metadata": {},
   "outputs": [],
   "source": [
    "rf_y_pred = RFregressor.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 202,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2270.784452116351"
      ]
     },
     "execution_count": 202,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mean_absolute_error(y_test, rf_y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 203,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "9708236.383523637"
      ]
     },
     "execution_count": 203,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mean_squared_error(y_test, rf_y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 204,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.6075613770954846"
      ]
     },
     "execution_count": 204,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "r2_score(y_test, rf_y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 205,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "RMSE of Linear Regression Model is  3115.8042915952915\n"
     ]
    }
   ],
   "source": [
    "from math import sqrt\n",
    "print(\"RMSE of Linear Regression Model is \",sqrt(mean_squared_error(y_test, rf_y_pred)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# XGBoost Regressor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 206,
   "metadata": {},
   "outputs": [],
   "source": [
    "from xgboost.sklearn import XGBRegressor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 207,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\Nantha\\Anaconda3\\lib\\site-packages\\xgboost\\core.py:587: FutureWarning: Series.base is deprecated and will be removed in a future version\n",
      "  if getattr(data, 'base', None) is not None and \\\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[23:14:35] WARNING: src/objective/regression_obj.cu:152: reg:linear is now deprecated in favor of reg:squarederror.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,\n",
       "             colsample_bynode=1, colsample_bytree=1, gamma=0,\n",
       "             importance_type='gain', learning_rate=1.0, max_delta_step=0,\n",
       "             max_depth=6, min_child_weight=40, missing=None, n_estimators=100,\n",
       "             n_jobs=1, nthread=None, objective='reg:linear', random_state=0,\n",
       "             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=0, silent=None,\n",
       "             subsample=1, verbosity=1)"
      ]
     },
     "execution_count": 207,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "xgb_reg = XGBRegressor(learning_rate=1.0, max_depth=6, min_child_weight=40, seed=0)\n",
    "\n",
    "xgb_reg.fit(X_train, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 208,
   "metadata": {},
   "outputs": [],
   "source": [
    "xgb_y_pred = xgb_reg.predict(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 209,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "2154.954637289423"
      ]
     },
     "execution_count": 209,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mean_absolute_error(y_test, xgb_y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 210,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "8290522.888761112"
      ]
     },
     "execution_count": 210,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "mean_squared_error(y_test, xgb_y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 211,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.664869986978737"
      ]
     },
     "execution_count": 211,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "r2_score(y_test, xgb_y_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 212,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "RMSE of Linear Regression Model is  2879.326811732408\n"
     ]
    }
   ],
   "source": [
    "from math import sqrt\n",
    "print(\"RMSE of Linear Regression Model is \",sqrt(mean_squared_error(y_test, xgb_y_pred)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The ML algorithm that perform the best was XGBoost Regressor Model with RMSE = 2879"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
