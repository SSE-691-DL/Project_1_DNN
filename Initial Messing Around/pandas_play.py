import pandas as pd
import numpy as np

# get version
p_version = pd.__version__
print('Pandas version is ' + (p_version))

# create series object
print('\n\nCreating Series object')
a = pd.Series(['Macon', 'Warrenton', 'Augusta'])
print(a)

# create dataframe object - missing values go NA/NaN
print('\n\nCreating DataFrame object')
print('City series')
empty_city = pd.Series(['Macon', 'Warrenton'])
print(empty_city)
city = pd.Series(['Macon', 'Warrenton', 'Augusta'])
print(city)
print('Pop series')
pseudo_pop = pd.Series([300000, 2000, 400000])
print(pseudo_pop)
print('\nEmpty cities DataFrame')
cities = pd.DataFrame({'City':empty_city, 'Pseudo-Population':pseudo_pop})
print(cities)
print('\nFilled cities DataFrame')
cities = pd.DataFrame({'City':city, 'Pseudo-Population':pseudo_pop})
print(cities)

# read csv into dataframe object
print('\n\nReading csv into DataFrame')
test_csv = pd.read_csv("test.csv", sep=",")
print('\nDetailed data analysis w/ describe')
print(test_csv.describe()) # detailed data analysis
print('\nFirst few records w/ head')
print(test_csv.head()) # first few records
print('\nHistogram of data (needs external tools to see)')
print(test_csv.hist('t_column_two')) # histogram of data

# access data
print('\n\nAccessing data')
print('\nType (series) followed by just cities')
print(type(cities['City']))
print(cities['City'])
print('\nType of first city followed by first city')
print(type(cities['City'][1]))
print(cities['City'][1])
print('\nType (DataFrame) followed by first and second entry')
print(type(cities[0:2]))
print(cities[0:2])

# data manipulation
print('\n\nData manipulation')
print('\nPop decreased by 3 OOM')
print(pseudo_pop / 1000)
print('\nLogarithmic values of Pop')
print(np.log(pseudo_pop)) # numpy takes series as input
print('\nData over 2000?')
# lambda function is applied to each value in series
print(pseudo_pop.apply(lambda v: v > 2000))
print('\nAdding columns')
cities['Area square miles'] = pd.Series([54, 90, 67])
cities['Population density'] = (cities['Pseudo-Population'] /
                                cities['Area square miles'])
print(cities)

print('\nCities who are large and start with M')
cities['Big and starts with M'] = ((cities['Area square miles'] > 50) &
                                   cities['City'].apply(lambda name:
                                                        name.startswith('M')))
print(cities)

# index properties
print('\nCity index values')
print(city.index)
print(city)
print('\nCities index values')
print(cities.index)
print(cities)

print('\n\nShuffling/randomizing DataFrame with reindex')
print('\nMy defined reindex of 0, 2, 1')
print(cities.reindex([0,2,1]))
print('\nRandom index values with numpy')
print(cities.reindex(np.random.permutation(cities.index)))
print('\nOOB reindexing')
print(cities.reindex([2,5,0]))
