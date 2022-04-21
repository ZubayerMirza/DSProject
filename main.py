import pandas as pd # Data Engineering
import matplotlib.pyplot as plt, seaborn as sns # Graph Visualizations
import folium, plotly.express as px # Choropleth Map Visualizations
import numpy as np # Quick Arrays
from folium.plugins import MarkerCluster # Clustering Markers on Choropleth Map
from sklearn.linear_model import LogisticRegression, LinearRegression
from scipy.special import expit
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Resources used:
#   Folium Map Documentation
#   Seaborn graph documentation
#   Interactive Visualization with Folium: https://medium.com/@saidakbarp/interactive-map-visualization-with-folium-in-python-2e95544d8d9b

# Data:
#   NYC Facilities Database: https://data.cityofnewyork.us/City-Government/Facilities-Database/ji82-xba5
#   DATA2GO.nyc
#   NYC Community Health Survey 2020: https://www1.nyc.gov/site/doh/data/data-sets/community-health-survey-public-use-data.page



# Part 1: Summary Statistics

# Read the sas file for Community Health Survey 2020
chs = pd.read_sas('dataset/chs2020_public.sas7bdat')
# Keep only needed features
chs = chs[['birthsex', 'imputed_povertygroup', 'didntgetcare20', 'skiprxcost', 'mood1', 'nspd',\
           'mood9', 'mood8', 'mhtreat20_all', 'mood11', 'delaypayrent', 'agegroup', 'emp3', \
           'imputed_neighpovgroup4_1519']]

# Use data where the person is in treatment for mental health or wants treatment
# Treatment include medication and therapy
chs['Mental Health'] = (chs['mhtreat20_all'] == 1) | (chs['nspd'] == 1) | (chs['mood11'] == 1) | (chs['mood1'] < 5)
sns.set(style="darkgrid")
color = ['g', 'r']

# SEX
sex = sns.countplot(x= 'birthsex', hue = 'Mental Health', data = chs, palette= color)
sex.legend(title= 'Mental Health',labels= ['Normal', 'Low'])
plt.title('Sex Distribution')
plt.xticks([0,1],['Male', 'Female'])

plt.savefig('graph/summary_stats/sex.png')

# POVERTY GROUP
plt.figure()
pov = sns.countplot(x = 'imputed_povertygroup', hue = 'Mental Health', data = chs, palette = color)
pov.legend(title= 'Mental Health', labels= ['Normal', 'Low'])
plt.title('Poverty Group Distribution')
plt.xticks(np.arange(5), ['<100%', '100-200%', '200-400%', '400-600%', '600%<'])
plt.xlabel('Poverty Group')
plt.savefig('graph/summary_stats/pov.png')

# INABILITY TO AFFORD PRESCRIPTION
plt.figure()
skip = sns.countplot(x = 'skiprxcost', hue = 'Mental Health', data = chs, palette= color)
skip.legend(title = "Mental Health", labels= ['Normal', 'Low'])
plt.title('Inability to Afford Medication')
plt.xticks(np.arange(2), ['Yes', 'No'])
plt.xlabel('Inability')
plt.savefig('graph/summary_stats/skip.png')

# NEIGHBORHOOD POVERTY LEVEL
plt.figure()
npov = sns.countplot(x ='imputed_neighpovgroup4_1519', hue = 'Mental Health', data = chs, palette= color)
npov.legend(title='Mental Health', labels= ['Normal', 'Low'])
plt.title('Neighborhood Poverty Level Distribution')
plt.xticks(np.arange(4), ['Low', 'Medium', 'High', 'Very High'])
plt.xlabel('Poverty Level')
plt.savefig('graph/summary_stats/npov.png')

# MISSED RENT
plt.figure()
delay = sns.countplot(x = 'delaypayrent', hue = 'Mental Health', data = chs, palette= color)
delay.legend(title='Mental Health', labels = ['Normal', 'Low'])
plt.title('Delaying to Pay Rent')
plt.xticks([0,1],['Yes', 'No'])
plt.xlabel('Delayed')
plt.savefig('graph/summary_stats/delay.png')

# EMPLOYMENT STATUS
plt.figure()
emp = sns.countplot(x = 'emp3', hue = 'Mental Health', data = chs, palette= color)
emp.legend(title= 'Mental Health', labels = ['Normal', 'Low'])
plt.title('Employment Distribution')
plt.xticks([0,1, 2],['Employed', 'Unemployed', 'Not in Labor Force'])
plt.xlabel('Employment Status')
plt.savefig('graph/summary_stats/emp.png')

# AGE
plt.figure()
age = sns.countplot(x = 'agegroup', hue = 'Mental Health', data = chs, palette= color)
age.legend(title='Mental Health', labels = ['Normal', 'Low'])
plt.title('Age Distribution')
plt.xticks([0, 1, 2, 3],['18 - 24', '25 - 44', '45 - 64', '65+'])
plt.xlabel('Age')
plt.savefig('graph/summary_stats/age.png')

# Part 2: MAP

# MAP 1: PSYCHIATRIC HOSPITALIZATIONS AND MENTAL HEALTH TREATMENT CENTERS

# Upload data2go data for map visualiztion
d2g = pd.read_csv('dataset/ALL_VARIABLES_D2G1.csv', low_memory= False)
d2g['GEO ID'] = d2g['GEO ID'].astype(str)
psych = d2g[['GEO ID', 'psychhosp_rate_cd']].dropna()
empl = d2g[['GEO ID', 'median_personal_earnings_puma']].dropna()
empl['GEO ID'] = empl['GEO ID'].str.slice(start=3)


# Upload second dataset of mental health facilities
fac = pd.read_csv('dataset/Facilities_Database.csv')
mh_fac =  fac[ fac['facsubgrp'] == 'MENTAL HEALTH' ].reset_index(drop=True)
emp_fac = fac[ fac['facsubgrp'] == 'WORKFORCE DEVELOPMENT' ].reset_index(drop=True)

# Mental Health and Psychiatic Hospitaliztion MAP

#Initialize the map and map clusters
mh_map = folium.Map(location=[40.7128, -74.006], zoom_start=11)

mh_cluster = MarkerCluster().add_to(mh_map)

# Add the map popups to marker cluster
for i in range(mh_fac.shape[0]):
    location = [mh_fac['latitude'][i],mh_fac['longitude'][i]]
    tooltip = "Zipcode:{}<br> Borough: {}<br> Click for more".format(mh_fac["postcode"][i], mh_fac['borough'][i])
    
    folium.Marker(location, # adding more details to the popup screen using HTML
                  popup="""
                  <i>Facility Name: </i> <br> <b>${}</b> <br> 
                  <i>Address: </i><b><br>{}</b><br>
                  <i>Facility Type: </i><b><br>{}</b><br>""".format(
                    mh_fac['facname'][i], 
                    mh_fac['address'][i], 
                    mh_fac['factype'][i]), 
                  tooltip=tooltip).add_to(mh_cluster)

# Adding the Chloropleth Map of Psychiatric Hospitalizations
folium.Choropleth(geo_data="geodata/community_districts.geojson",
                     fill_color='YlOrRd',
                     data = psych,
                     key_on='feature.properties.boro_cd',
                     columns = ['GEO ID', 'psychhosp_rate_cd'],
                     legend_name= 'Psychiatric Hospitalizations'
                     ).add_to(mh_map)

mh_map.save('graph/choropleth/mh_map.html')


# MAP 2: Median Income by City District and Employment Service Centers

# Create map and map clusters
emp_map = folium.Map(location=[40.7128, -74.006], zoom_start=11)

emp_cluster = MarkerCluster().add_to(emp_map)

# Add each map popup to the map cluster
for i in range(emp_fac.shape[0]):
    location = [emp_fac['latitude'][i],emp_fac['longitude'][i]]
    tooltip = "Zipcode:{}<br> Borough: {}<br> Click for more".format(emp_fac["postcode"][i], emp_fac['borough'][i])
    
    folium.Marker(location, # adding more details to the popup screen using HTML
                  popup="""
                  <i>Facility Name: </i> <br> <b>${}</b> <br> 
                  <i>Address: </i><b><br>{}</b><br>
                  <i>Facility Type: </i><b><br>{}</b><br>""".format(
                    emp_fac['facname'][i], 
                    emp_fac['address'][i], 
                    emp_fac['factype'][i]), 
                  tooltip=tooltip).add_to(emp_cluster)

#Add the Choropleth map of median income by city district
folium.Choropleth(geo_data="geodata/puma.json",
                     fill_color='YlGn',
                     data = empl,
                     key_on='feature.properties.PUMA',
                     columns = ['GEO ID', 'median_personal_earnings_puma'],
                     legend_name= 'Median Personal Earnings'
                     ).add_to(emp_map)
folium.LayerControl().add_to(emp_map)

emp_map.save(outfile='graph/choropleth/emp_map.html')

# Part 3: Model Prediction Logistic Regression

plt.figure()
sns.regplot(x='imputed_povertygroup', y='Mental Health', data=chs, x_jitter = .3, y_jitter = .05, logistic=True)
plt.title("Linear Regression of Poverty Group vs Mental Health")
plt.yticks([0,1], ['Normal', 'Low'])
plt.xticks(np.arange(1,6), ['<100%', '100-200%', '200-400%', '400-600%', '600%<'])


# Don't have strong probabilities in either direction

plt.savefig('graph/predictive_model/povlog.png')

# Confusion Matrix of Poverty Group and Mental Health

X = chs['imputed_povertygroup'] # Independent Variable
y = chs['Mental Health'] # Dependent Variable to predict

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train = X_train.values.reshape(-1,1)
X_test = X_test.values.reshape(-1,1)

clf = LogisticRegression(C=1e5)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

# Confusion matrix
cf_matrix = confusion_matrix(y_test, y_pred)
plt.figure()
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues', xticklabels=['Normal', 'Low'], yticklabels=['Normal', 'Low'])
stat_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1)
plt.xlabel('Predictor')
plt.ylabel('Predicted')
plt.title('Poverty Group Heat Map')
plt.savefig('graph/predictive_model/povheatmap.png')