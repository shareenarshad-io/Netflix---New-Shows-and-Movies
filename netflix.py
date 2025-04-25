#Netflix Dataset from Stratascratch - Assignment answers & walkthrough
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

# The full end-to-end ML lifecycle does not apply to this, first data pre-processing was applied in order to clean the data and explore the data to make it better useable for analysis 
# 1. Problem Definition - described in README
# 2. Data Collection and Preparation - provided in .csv file, completed before answering questions
# 3. Exploratory Data Analysis - completed before answering questions
# 4. Model Building
# 5. Model Evaluation 
# 6. Deployment and Communication. 

# Read a CSV file
df = pd.read_csv('netflix_data.csv')
print(df)

#Data Preparation and Exploratory Data Analysis 

##Answers to Questions from Assignment 

#1. What type of content is available in different countries?
# Best way to answer is to create a bar chart with country and type included in it 
df1 = df.take([1,5], axis=1)
print(df1)
distinct_count = df1['country'].nunique()
print(distinct_count)

content_by_country = df.groupby(["country", "type"]).size().unstack().fillna(0)
content_by_country.sort_values(by="Movie", ascending=False, inplace=True)

plt.figure(figsize=(12, 6))
content_by_country.head(10).plot(kind="bar", stacked=True)
plt.title("Top 10 Countries by Content Type on Netflix")
plt.xlabel("Country")
plt.ylabel("Number of Titles")
plt.legend(title="Type")
plt.xticks(rotation=45)
plt.show()

#2. How has the number of movies released per year changed over the last 20-30 years?



#3. Comparison of tv shows vs. movies.

#4. What is the best time to launch a TV show?

#5. Analysis of actors/directors of different types of shows/movies.

#6. Does Netflix has more focus on TV Shows than movies in recent years?

#7. Understanding what content is available in different countries.