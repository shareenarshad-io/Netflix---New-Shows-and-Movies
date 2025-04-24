#Netflix Dataset from Stratascratch - Assignment answers & walkthrough
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

# The full end-to-end ML lifecycle does not apply to this, first data pre-processing was applied in order to clean the data and explore the data to make it better useable for analysis 

# Read a CSV file
df = pd.read_csv('netflix_data.csv')
print(df)

'''
#Data preprocessing steps:
# Check for missing values
missing_values = df.isnull().sum().sort_values(ascending=False)
missing_values = missing_values[missing_values > 0]  # Only show columns with missing values

# Save missing values plot
plt.figure(figsize=(10, 5))
missing_values.plot(kind="bar", color="red", alpha=0.8)
plt.title("Missing Values per Column")
plt.xlabel("Columns")
plt.ylabel("Count of Missing Values")
plt.xticks(rotation=45)
plt.show()

# Fill missing values with appropriate defaults
#df["date_added"] = pd.to_datetime(df["date_added"], errors="coerce")
#df["date_added"].fillna(pd.Timestamp("2000-01-01"), inplace=True)


df = df.assign(
    rating=df["rating"].fillna("Unknown"),
    country=df["country"].fillna("Unknown"),
    cast=df["cast"].fillna(""),
    director=df["director"].fillna(""),
    listed_in=df["listed_in"].fillna("")
)


# Extract year and month from date_added
#df["month_added"] = df["date_added"].at.month
#df["year_added"] = df["date_added"].at.year

# Convert release_year to numeric
df["release_year"] = pd.to_numeric(df["release_year"], errors="coerce")

# Summary statistics for numerical columns
numeric_summary = df.describe()
numeric_summary.to_csv("graphs/numeric_summary.csv")

# Save numeric summary visualization
plt.figure(figsize=(10, 5))
sns.boxplot(data=df[["release_year"]].dropna())
plt.title("Distribution of Release Years")
plt.show()

# Extract main genre from listed_in
df["main_genre"] = df["listed_in"].apply(lambda x: x.split(",")[0] if isinstance(x, str) else x)
'''



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