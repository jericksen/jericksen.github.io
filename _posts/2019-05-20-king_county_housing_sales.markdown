---
layout: post
title:      "King County Housing Sales"
date:       2019-05-20 20:57:44 -0400
permalink:  king_county_housing_sales
---



Original Posted on Medium: https://medium.com/@jericksen20/king-county-real-estate-14b58484798c



A recent linear regression modeling project required a detailed exploration of a dataset containing housing sales data from King County, WA. The dataset housed multiple variables that included the sale price of the property, when the property was sold, and various other data describing the property’s attributes. Additionally, the dataset included geographic data as well as information on the immediate surrounding properties.

I began this project with some exploratory analysis seeking to answer these three questions:

How are housing prices (home values) distributed throughout King County, WA? And what factors might explain the concentration of higher housing values in the northeast corner of the county?
Does the age of the house have a noticeable impact on value?
Does King County exhibit any seasonality in terms of housing inventory turnover?
For this project, I begin by importing multiple packages including the pandas as well as seaborn for use within a Jupyter Notebook (note: some packages were used for the modeling aspect of this project):

- Import applicable libraries/packages
import pandas as pd
import pandas_profiling
import numpy as np
import statsmodels.api as sm
import statsmodels.stats as sts
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
Using pandas, I import the dataset as a data frame and assigned it to the variable ‘df’:

df = pd.read_csv('kc_house_data.csv')
From there, I scrubbed the dataset of null values, unnecessary attributes (columns), and replaced missing values with the attribute medians:

- Removing null values
df.dropna(how = 'any', axis = 0, subset = ['view'], inplace=True)
- Removing columns
df.drop(columns = ['yr_renovated','id', 'sqft_lot15', 'condition', 'sqft_lot'], axis=1, inplace=True)
- Replacing missing values
df.waterfront = df.waterfront.fillna(value=df.waterfront.median)
In approaching the answer to the first question, I decided to produce the hexbin visual with seaborn which plots housing sales by price using the coordinates provided in the dataset. Below is an image of the hexbin graphic along with the code:


df.plot.hexbin(x='long', y='lat', C='price', gridsize=60, cmap='coolwarm', figsize=(11, 9))
plt.title('Location By Sale Price', fontsize = 15)
plt.ylabel('Latitude', fontsize = 12)
plt.xlabel('Longitude', fontsize = 12)
plt.show()
Below, a screenshot of King County from google maps was imported to assist with determining key features that might explain the variation in housing values. This image was imported using the following code:


from IPython.display import Image
Image(filename = 'King County Metro Area.png', width=400, height=400)
Based on these visuals, there appears to be a high concentration of higher housing values in and around the Kirkland area.This is likely explained due to it’s proximity to highly concentrated white collar work environments, close proximity to shopping, upscale eateries, et cetera.

Additionally, housing prices seem further exacerbated by proximity to the popular Lake Washington. As per convention, lakeside housing prices are often correlated with higher prices based on quality water views, access to beaches, and recreation.

It’s likely these are the two largest contributing factors with respect to higher housing in and around the Kirkland area.

To answer the second question, I chose to plot the housing price against the year the property using seaborn’s lmplot. By doing so, I was able to witness any correlation that might exist between property age and sale price. Additionally, lmplots append a histogram for each variable along with the Kernel Density line opposing the axis outside the scatter plot:


sns.jointplot(x='yr_built', y='price' , data=df, kind = 'reg', height = 10, xlim=(1895, 2020), color = 'g', ratio = 7)
plt.title('Sale Price by Year Built')
plt.ylabel('Sell Price', fontsize = 14)
plt.xlabel('Year Built', fontsize = 14)
plt.show()
With this, it was clear age has little to no effect on the on sale price thus answering our second question: Does age affect housing prices?

In approaching the final question, I decided a simple histogram would suffice. The question asked was whether or not inventory turnover varies by month, i.e., does King County exhibit seasonality with respect to inventory turnover? Below is the histogram plotting home sales by month along with the code used to plot the graphic:


plt.figure(figsize = (13,8))
ax = df["month"].value_counts().sort_index().plot(kind='bar')plt.title("Home Sales By Month", fontsize = 20)
plt.ylabel('Total Home Sales', fontsize = 15)
plt.xlabel('Month', fontsize = 15)
plt.xticks(ticks=range(12), labels=months)
plt.show()
From the histogram above, it’s clear inventory turnover is lowest during the winter months with January having less than 1,000 closings throughout King County. The month of May had the highest turnover with roughly 2,400 closings. That’s more than double housing sales from January. More broadly, the spring and summer (March — August) months saw much higher turnover than the fall and winter months (September — February).

Seasonal variation is common in sales cycles across many industries. King County appears to be no exception. In order to confirm this cycle is a regular occurrence, we would need housing data for additional years in order to have confidence in our initial conclusion of seasonality.

In conclusion, EDA is an important process during the data science process as it can answer many questions through simple visualizations. Building data visualizations affords us the luxury of understanding our data with minimal cognitive input. This, of course, stems from our visual cortex being a powerful engine for understanding the world around us.



