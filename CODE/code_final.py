# -*- coding: utf-8 -*-
# Import libraries
import pip
import tkinter as tk
from tkinter import *
from tkinter import *
from tkinter import ttk
import pandas as pd
import numpy as np
import os
import pickle
import webbrowser
from io import BytesIO
import requests
try:
    import xgboost as xgb
except ImportError:
    pip.main(['install', 'xgboost'])
    import xgboost as xgb
try:
    import geopandas as gpd
except ImportError:
    pip.main(['install', 'geopandas'])
    import geopandas as gpd
# Install keplergl library
try:
    from keplergl import KeplerGl
except ImportError:
    pip.main(['install', 'keplergl'])
    from keplergl import KeplerGl

# Get working directory
directory = os.getcwd()

mfile = directory + '/trained_model.pkl'
loaded_model = pickle.load(open(mfile, 'rb'))

# Import the dataset
# Processed raw Data
url = 'https://raw.githubusercontent.com/yanzhegeo/airdelay/main/known.csv'
df_known = pd.read_csv(url, index_col=0)

# import GlobalAirportDatabase.txt from github and preprocess
url = 'https://raw.githubusercontent.com/yanzhegeo/airdelay/main/GlobalAirportDatabase.txt'
df_coord = pd.read_csv(url, header=None)

column_name = ['ICAO', 'IATA', 'Airport', 'City', 'Country', '1', '1', '1', '1', '1', '1', '1', '1', 'Altitude',
               'Latitude', 'Longitude']
df_coord[column_name] = df_coord[0].str.split(':', expand=True)

# Filter US airports only for airport dataset
df_coord = df_coord[df_coord.Country == "USA"]
df_coord = df_coord[["IATA", "City", "Latitude", "Longitude"]]
df_coord = df_coord[(df_coord.IATA != "N/A") & (df_coord.Latitude != "0.000")].reset_index(drop=True)

# import map configurations from github
url_configall = 'https://raw.githubusercontent.com/yanzhegeo/airdelay/main/Configs/config_all.csv'
config_all = pd.read_csv(url_configall, index_col=[0])

# Calculate the great circle distance between two points
def dist_calc(origin, dest):
    # Get coordinates from airport df
    lon1 = float(df_coord.loc[df_coord['IATA'] == origin].iloc[0]['Longitude'])
    lat1 = float(df_coord.loc[df_coord['IATA'] == origin].iloc[0]['Latitude'])
    lon2 = float(df_coord.loc[df_coord['IATA'] == dest].iloc[0]['Longitude'])
    lat2 = float(df_coord.loc[df_coord['IATA'] == dest].iloc[0]['Latitude'])
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    # Get difference in lat and lon
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    # trig calculation
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km

# Generate statistical results and show in choices buttons
'''
Function: Map by Origin/Dest, and filter by top 30
Input: od: deafault:'Origin'
           'Origin': map by origins, 'Dest': map by destinations
       airport: default: 'ATL' 
               airport: this is either the origin or the destination of your chosen path
'''
def delay_map(od='Origin', airport='ATL'):
    if airport in set(df_coord.IATA):
        print('IN')
        df = df_known.loc[df_known[od] == airport]
        if od == 'Origin':
            od = 'Dest'
        elif od == 'Dest':
            od = 'Origin'
        try:
            arr_delay_od = df.groupby(by=[od]).agg({"Delay_YN": "mean", "ArrDelay": "median"}).reset_index()
        except:
            print("argument should be 'Origin' or 'Dest' only ")

    else:
        print('We do not have coordinates for this airport')
        try:
            arr_delay_od = df_known.groupby(by=[od]).agg({"Delay_YN": "mean", "ArrDelay": "median"}).reset_index()
        except:
            print("argument should be 'Origin' or 'Dest' only ")

    od_table = arr_delay_od.sort_values(["Delay_YN"], ascending=(False))
    od_table['Delay_YN'] = (100. * od_table['Delay_YN']).round(1).astype(float)
    od_table = od_table.rename(columns={'Delay_YN': 'Delay Chance/%', 'ArrDelay': 'Delay Time/mins'})

    # join airport location data to delay predictions
    df_od = od_table.merge(df_coord, how="left", left_on=od, right_on="IATA")
    # Load saved config file
    this_config = eval(config_all.iloc[0][0])
    df_od = df_od[df_od['IATA'].notna()]

    # Create a basemap
    map = KeplerGl(height=600, width=900)

    # Create a gepdataframe and convert to geojson
    gdf = gpd.GeoDataFrame(df_od, geometry=gpd.points_from_xy(df_od.Longitude, df_od.Latitude))
    gdf.to_file("od_json.geojson", driver="GeoJSON")
    with open('od_json.geojson', 'r') as f:
        geojson = f.read()
    # Add geojson data to Kepler
    map.save_to_html(data={'airports': geojson}, config=this_config, file_name='this_map.html')
    # Show map in browser
    webbrowser.open('this_map.html')

# Import prediction data
final = pd.read_csv(directory + '/final.csv')
final_predictors = pd.read_csv(directory + '/final_predictors.csv')

# Setup default inputs for model output to display
zero_data = np.zeros(shape=(1, len(final_predictors.columns)))
default = pd.DataFrame(zero_data, columns=final_predictors.columns)
default['CRSDepTime'] = final_predictors['CRSDepTime'].mean()
default['DayofMonth'] = final_predictors['DayofMonth'].mean()
default['Distance'] = final_predictors['Distance'].mean()

month = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6, 'July': 7, 'August': 8,
         'September': 9, 'October': 10, 'November': 11, 'December': 12}
week = {'Sunday': 1, 'Monday': 2, 'Tuesday': 3, 'Wednesday': 4, 'Thursday': 5, 'Friday': 6, 'Saturday': 7}
time = {'Early Morning (0:00-4:30)': 7, 'Morning (4:30-8:00)': 1, 'Late Morning (8:00-10:00)': 2,
        'Midday (10:00-12:00)': 3, 'Afternoon (12:00-15:00)': 4, 'Evening (15:00-17:30)': 5, 'Night (17:30-20:30)': 6,
        'Late Night (20:30-0:00)': 0}
origin = list(final.Origin.unique())
origin.sort()
dest = list(final.Dest.unique())
dest.sort()
airline = list(final.Description.unique())
airline.sort()

# Predict delay of user input based on the trained xgb model
def predict(output):
    newpt = default.copy()
    newpt['Month_' + str(month[output[0]])] = 1
    newpt['DOW_' + str(week[output[1]])] = 1
    newpt['Origin_' + output[2]] = 1
    newpt['Dest_' + output[3]] = 1
    newpt['Description_' + output[4]] = 1
    newpt['Time_' + str(time[output[5]])] = 1
    # Calculate the path distance if we know their coordinates
    try:
        newpt['Distance'] = dist_calc(output[2], output[3])
    except Exception:
        pass
    pred = loaded_model.predict(xgb.DMatrix(newpt))
    print('Delay Probability: ' + str(pred[0]))
    return pred[0]

# User interactive interface.
master = Tk()

# 1. month label and variable
monthVar = StringVar(master)
monthVar.set(list(month.keys())[0])  # default value
w = OptionMenu(master, monthVar, *month.keys())
label1 = Label(master, text='Month:')
label1.grid(row=1, column=0)
w.grid(row=2, column=0)

# 2. week label and variable
weekVar = StringVar(master)
weekVar.set(list(week.keys())[0])  # default value
w2 = OptionMenu(master, weekVar, *week.keys())
label2 = Label(master, text='Day Of Week:')
label2.grid(row=3, column=0)
w2.grid(row=4, column=0)

# 3. origin/departure label and variable
# type in and then choose from the searchbox for origin
def scan_origin(event):
    val = event.widget.get()
    if val == '':
        data = origin
    else:
        data = []
        for item in origin:
            if val.lower() in item.lower():
                data.append(item)
    update_origin(data)

def update_origin(data):
    listbox_origin.delete(0, 'end')
    # put new data
    for item in data:
        listbox_origin.insert('end', item)

label3 = Label(master, text='Origin:')
label3.grid(row=1, column=1)
entry_origin = Entry(master)
entry_origin.grid(row=2, column=1)
entry_origin.bind('<KeyRelease>', scan_origin)

listbox_origin = Listbox(master,exportselection=0)
update_origin(origin)
listbox_origin.select_set(0)
listbox_origin.grid(row=2,rowspan=8, column=1)

# 4. dest/arrival label and variable
# type in and then choose from the searchbox for destination
def scan_dest(event):
    val = event.widget.get()
    if val == '':
        data = dest
    else:
        data = []
        for item in origin:
            if val.lower() in item.lower():
                data.append(item)
    update_dest(data)

def update_dest(data):
    listbox_dest.delete(0, 'end')
    # put new data
    for item in data:
        listbox_dest.insert('end', item)
label4 = Label(master, text='Destination:')
label4.grid(row=1, column=2)
entry_dest = Entry(master)
entry_dest.grid(row=2, column=2)
entry_dest.bind('<KeyRelease>', scan_dest)

listbox_dest = Listbox(master,exportselection=0)
update_dest(dest)
listbox_dest.selection_set(1)
listbox_dest.grid(row=2,rowspan=8, column=2)

# 5. airline label and variable
airlineVar = StringVar(master)
airlineVar.set(airline[0])  # default value
w5 = OptionMenu(master, airlineVar, *airline)
label5 = Label(master, text='Airline:')
label5.grid(row=5, column=0)
w5.grid(row=6, column=0)

# 6. time of the day label and variable
timeVar = StringVar(master)
timeVar.set(list(time.keys())[0])  # default value
w6 = OptionMenu(master, timeVar, *time.keys())
label6 = Label(master, text='Time of Day:')
label6.grid(row=7, column=0)
w6.grid(row=8, column=0)

# 7. Do you want to map all flights from the origin or all flights to the destination
map_choices = ['Origin', 'Destination']
mapVar = StringVar(master)
mapVar.set(map_choices[0])  # default value
w7 = OptionMenu(master, mapVar, *map_choices)
label7 = Label(master, text='Map by:')
label7.grid(row=9, column=1)
w7.grid(row=10, column=1)

# Prediction and Mapping button with functions
def ok():
    # Calculate Model predictors
    departure = str(listbox_origin.get(listbox_origin.curselection()))
    arrival = str(listbox_dest.get(listbox_dest.curselection()))
    name = str(timeVar.get())
    output = [monthVar.get(), weekVar.get(), departure, arrival, airlineVar.get(), timeVar.get()]
    predd = predict(output)
    # Mapping function here
    if mapVar.get() == 'Destination':
        delay_map(od='Dest', airport=arrival)
    else:
        delay_map(od='Origin', airport=departure)

    model_output = 1

    window = Tk()

    window.title("Team51.io")

    tab_control = ttk.Notebook(window)

    tab1 = ttk.Frame(tab_control)

    # PREDICTIONS FROM MODEL
    tab_control.add(tab1,
                    text="""

    - Airport Locations -

    Departure Location: %s

    Arrival Location: %s



    - Delay Prediction Confidence -

    %s :  %s         


    Thank you for choosing Team51.io to help you pick your flight!!   
    """ % (departure, arrival, name, str(round(predd * 100)) + "%"))

    tab_control.pack(expand=1, fill='both')

    window.mainloop()

# Button initialization
button = Button(master, text="OK", command=ok)
button.grid(row=11, columnspan=3)

# Show the tkinter
mainloop()
