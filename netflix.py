#Netflix Dataset - Assignment answers & walkthrough

# The full end-to-end ML lifecycle does not apply to this, first data pre-processing was applied in order to clean the data and explore the data to make it better useable for analysis 
# 1. Problem Definition - described in README
# 2. Data Collection and Preparation - provided in .csv file, completed before answering questions
# 3. Exploratory Data Analysis - completed before answering questions
# 4. Model Building
# 5. Model Evaluation 
# 6. Deployment and Communication. 

#Import libraries 
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from itertools import chain
import seaborn as sns

# Read a CSV file
df = pd.read_csv('netflix_data.csv')
print(df.head())

##Answers to Questions from Assignment 

#1. What type of content is available in different countries?
# Best way to answer is to create a bar chart with country and type included in it

df['country'] = df['country'].fillna('Unknown')

top_countries = df['country'].value_counts().head(10)

plt.figure(figsize=(10, 6))
top_countries.plot(kind='bar', color='tomato')
plt.title('Top 10 Countries with Most Netflix Content')
plt.ylabel('Number of Titles')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#2. How has the number of movies released per year changed over the last 20-30 years?

df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
movies = df[df['type'] == 'Movie']
movies_per_year = movies.groupby('release_year').size()
movies_per_year = movies_per_year[movies_per_year.index >= 1995]
print(movies_per_year.head())

plt.figure(figsize=(12, 6))
movies_per_year.plot(kind='line', marker='o')
plt.title('Movies Released Per Year (Since 1995)')
plt.xlabel('Year')
plt.ylabel('Number of Movies')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


#3. Comparison of tv shows vs. movies.
# Best to use a pie chart for this 

type_counts = df['type'].value_counts()

plt.figure(figsize=(6, 6))
type_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['skyblue', 'lightgreen'])
plt.title('TV Shows vs. Movies on Netflix')
plt.ylabel('')
plt.tight_layout()
plt.show()

#4. What is the best time to launch a TV show?
import pandas as pd
import matplotlib.pyplot as plt

df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
df['month_added'] = df['date_added'].dt.month
tv_shows = df[df['type'] == 'TV Show']
tv_by_month = tv_shows['month_added'].value_counts().sort_index()

month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

plt.figure(figsize=(10, 6))
plt.bar(month_names, tv_by_month.values, color='skyblue')
plt.title('Netflix TV Show Releases by Month', fontsize=14)
plt.xlabel('Month')
plt.ylabel('Number of TV Shows Added')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


#5. Analysis of actors/directors of different types of shows/movies.

df['cast'] = df['cast'].fillna('')
actor_counts = Counter(chain.from_iterable([x.split(', ') for x in df['cast'] if x]))
top_actors = dict(actor_counts.most_common(10))

plt.figure(figsize=(10, 6))
plt.bar(top_actors.keys(), top_actors.values(), color='goldenrod')
plt.title('Top 10 Most Common Actors on Netflix')
plt.xlabel('Actor')
plt.ylabel('Number of Appearances')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
#6. Does Netflix has more focus on TV Shows than movies in recent years?

df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
content_by_year = df.groupby(['release_year', 'type']).size().unstack().fillna(0)
recent_content = content_by_year[content_by_year.index >= 2010]

plt.figure(figsize=(12, 6))
recent_content.plot(kind='line', marker='o')
plt.title('Trend of TV Shows vs Movies Over Time (Since 2010)')
plt.xlabel('Release Year')
plt.ylabel('Number of Titles')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

#7. Understanding what content is available in different countries.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = df[df['country'].notna() & (df['country'].str.strip() != '')]
df = df[~df['country'].str.lower().isin(['unknown'])]

df['country'] = df['country'].str.strip()
df['type'] = df['type'].str.strip().str.title()

country_type_counts = df.groupby(['country', 'type']).size().unstack().fillna(0)

filtered = country_type_counts[(country_type_counts['Movie'] > 0) & (country_type_counts['Tv Show'] > 0)]

top5 = filtered.sort_values(by='Movie', ascending=False).head(5)

labels = top5.index.tolist()
movie_counts = top5['Movie'].tolist()
tv_counts = top5['Tv Show'].tolist()

x = np.arange(len(labels)) 
width = 0.35  

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, movie_counts, width, label='Movie', color='skyblue')
bars2 = ax.bar(x + width/2, tv_counts, width, label='TV Show', color='orange')

ax.set_xlabel('Country')
ax.set_ylabel('Number of Titles')
ax.set_title('Content Availability by Country (Top 5) – Grouped Bar Chart')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45)
ax.legend()
ax.bar_label(bars1, padding=3)
ax.bar_label(bars2, padding=3)
plt.tight_layout()
plt.show()