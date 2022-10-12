import csv
import pandas as pd
import matplotlib.pyplot as plt

rows = []

with open("total_stars.csv", "r") as f:
    csvreader = csv.reader(f)
    for row in csvreader:
        rows.append(row)

headers = rows[0]
star_data_rows = rows[1:]

print(headers)
print(star_data_rows)

headers[0] = "row_num"

temp_stars_data_rows = list(star_data_rows)
for star_data in temp_stars_data_rows:
    star_mass = star_data[3]
    if star_mass.lower() == "unknown":
        star_data_rows.remove(star_data)
        continue
    else:
        star_mass_value = star_mass.split("")[0]
        star_mass_ref = star_mass.split("")[1]
        if star_mass_ref == "Icarus":
            star_mass_value = float(star_mass_value) * 1.989e+30
        star_data[3] = star_mass_value

    star_radius = star_data[4]   
    if star_radius.lower() == "unknown":
        star_data_rows.remove(star_data)
        continue
    else:
        star_radius_value = star_radius.split("")[0]
        star_radius_ref = star_radius.split("")[2]
        if star_radius_ref == "Icarus":
            star_radius_value = float(star_radius_value) * 6.957e+8
        star_data[4] = star_radius_value 

print(len(star_data_rows))

import plotly.express as px

hd_10180_star_masses = []
hd_10180_star_names = []
for star_data in star_data_rows:
    hd_10180_star_masses.append(star_data[3])
    hd_10180_star_names.append(star_data[1])

hd_10180_star_masses.append(1)
hd_10180_star_names.append("sun")

fig = px.bar(x = hd_10180_star_names, y = hd_10180_star_masses)
fig.show()

temp_stars_data_rows = list(star_data_rows)
for star_data in temp_stars_data_rows:
    if star_data[1].lower() == "hd1 00546 b":
        star_data_rows.remove(star_data)

star_masses =[]
star_radiuses = []
star_names = []
for star_data in star_data_rows:
    star_masses.append(star_data[3])
    star_radiuses.append(star_data[4])
    star_names.append(star_data[1])
star_gravity = []
for index,  name in enumerate(star_names):
    gravity = (float(star_masses[index])*5.972e+24) / (float(star_radiuses[index])*float(star_radiuses[index])*6371000*6371000) * 6.674e-11
    star_gravity.append(gravity)
fig  = px.scatter(x = star_radiuses, y = star_masses, size = star_gravity, hover_data = [star_names]) 
fig.show()

low_gravity_star = []
for index, gravity in enumerate(star_gravity):
    if gravity < 100:
        low_gravity_star.append(star_data_rows[index])
print(len(low_gravity_star))

print(headers)

star_masses = []
star_radiuses = []
for star_data in low_gravity_star:
    star_masses.append(star_data[3])
    star_radiuses.append(star_data[4])
fig = px.scatter(x = star_radiuses, y = star_masses) 
fig.show()

from sklearn.cluster import Kmeans
import matplotlib.pyplot as plt
import seaborn as sns

X = []
for index, star_mass in enumerate(star_masses):
    temp_list = [
        star_radiuses[index],
        star_mass
    ]
    X.append(temp_list)

wcss = []
for i in range(1, 11):
    kmeans = Kmeans(n_clusters=i, init='k-means++', random_state = 42) 
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)   

plt.figure(figsize=(10,5))
sns.lineplot(range(1, 11), wcss, marker='o', color='red')
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
    






