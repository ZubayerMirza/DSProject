import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import plotly.express as px
import json

# Part 1: Summary Statistics

# Read the sas file for Community Health Survey 2020
chs = pd.read_sas('chs2020_public.sas7bdat')

# Feature Engineer an isdepressed column based off the the mood level. Level 5 is absolutely no depression

chs = chs[['birthsex', 'imputed_povertygroup', 'didntgetcare20', 'skiprxcost', 'mood1', 'nspd',\
     'mood9', 'mood8', 'mhtreat20_all', 'mood11', 'delaypayrent', 'agegroup', 'emp3', \
         'imputed_neighpovgroup4_1519']]
chs['isdepressed'] = chs['mood1'].apply(lambda x: 'Depressed' if x <= 4 else 'Not Depressed')
chs['sex'] = chs['birthsex'].apply(lambda x: 'Male' if x == 1 else 'Female')

sex = sns.countplot(x= 'isdepressed', hue= 'sex', data = chs)
plt.title('Sex vs Mood')
plt.savefig('graph/sex.png')

fpie = plt.pie()

plt.figure()
sns.countplot(x="isdepressed", hue= 'imputed_povertygroup', data=chs)
plt.title('Poverty Level vs Mood')
plt.savefig('graph/econ.png')

plt.figure()
sns.countplot(x="isdepressed", hue= 'didntgetcare20', data=chs)
plt.title('Did not recieve care vs Mood')
plt.savefig('graph/nocare.png')

plt.figure()
sns.countplot(x="isdepressed", hue= 'weightall', data=chs)
plt.title('Weight vs Mood')
plt.savefig('weight.png')

# Part 2: Chloropleth Map from d2g data
d2g = pd.read_csv('ALL_VARIABLES_D2G1.csv', low_memory= False)
psych = d2g[['GEO ID', 'psychhosp_rate_cd']].dropna()
psych['GEO ID'] = psych['GEO ID'].astype(str)
empl = d2g[['GEO ID', 'median_personal_earnings_puma']].dropna()
empl['GEO ID'] = empl['GEO ID'].astype(str).str.slice(start=3)


map = folium.Map(location=[40.75, -74.125])

folium.Choropleth(geo_data="community_districts.geojson",
                     fill_color='YlOrRd',
                     data = psych,
                     key_on='feature.properties.boro_cd',
                     columns = ['GEO ID', 'psychhosp_rate_cd'],
                     legend_name= 'Psychiatric Hospitalizations'
                     ).add_to(map)
folium.LayerControl().add_to(map)
map.save(outfile='map.html')

cd = json.load(open("community_districts.geojson", 'r'))

fig = px.choropleth(psych, geojson = cd, locations='GEO ID', color= 'psychhosp_rate_cd',
                           color_continuous_scale=px.colors.sequential.Plasma,
                           featureidkey= 'properties.boro_cd'
                          )
fig.update_layout(title_text = 'Psychiatric Hospitalizations in New York City', title_x = 0.5,)
fig.update_geos(fitbounds = 'locations')
fig.show()

map2 = folium.Map(location=[40.75, -74.125])
folium.Choropleth(geo_data="puma.json",
                     fill_color='YlOrRd',
                     data = empl,
                     key_on='feature.properties.PUMA',
                     columns = ['GEO ID', 'median_personal_earnings_puma'],
                     legend_name= 'Median Personal Earnings'
                     ).add_to(map2)
folium.LayerControl().add_to(map2)
map2.save(outfile='map2.html')

puma = json.load(open("puma.json", 'r'))
fig2 = px.choropleth(empl, geojson = puma, locations='GEO ID', color= 'median_personal_earnings_puma',
                           color_continuous_scale=px.colors.sequential.YlGn,
                           featureidkey= 'properties.PUMA'
                          )
fig2.update_layout(title_text = 'Mean Personal Income', title_x = 0.5,)
fig2.update_geos(fitbounds = 'locations')
fig2.show()

hosp = pd.read_excel('HEALTH_D2G.xlsx')
work = pd.read_excel('WORK_WEALTH_POVERTY_D2G.xlsx')

health = chs[['weightall', 'mood1']]
smoker = [['smoker', 'mood1']]
diff_living = [['difficultdailyact', 'mood1']]

# Part 3 Logistic Regression

chs['has_depression'] = chs['isdepressed'] = chs['mood1'].apply(lambda x: 1 if x <= 4 else 0)
econ = chs[['imputed_povertygroup', 'mood1', 'has_depression']]

plt.figure()
sns.regplot(x='imputed_povertygroup', y='has_depression', data=econ, x_jitter=.1, y_jitter = .1)
plt.title("Linear Regression Plot of Poverty Group vs Depression")
plt.savefig('linearplot.png')

