DEMO
A demonstration of how to install and execute our code can be found here:
https://www.youtube.com/watch?v=nfghWC0h5C8 


1. DESCRIPTION
The objective of this application is to provide travellers a tool to mitigate the adverse effects of delays by providing them accurate and customized predictions of experiencing a significant flight delay for them to use in their travel planning. To do this, our application provides travellers with predictions about the likelihood of significant delays for flights given origin, destination, carrier, date, and time. In this application, significant delay is defined as a delay greater than 30 minutes. 

The application consists of an interactive map designed in kepler.gl, an interactive dropdown menu using tkinter, and xgBoost predictive model. In the interactive map, users can hover over airports in the map to display baseline rates of significant delay and median delay time. Airports with longer delay times are shown with bigger circles. Airports with higher rates of significant delay are darker in color. 

Data used in the predictive model comes from https://www.kaggle.com/bulter22/airline-data. 

2. INSTALLATION
Before running final_code.py, ensure the following libraries are installed:
tkinter
pandas
numpy
xgboost
os
pickle
io
requests
keplergl
geopandas
webbrowser

3.EXECUTION
-Unzip the ‘CODE’ folder. 
-Open terminal and navigate to the ‘CODE’ folder. 
-In the terminal, start a python session then open ‘final_code.py’.  
-A separate window should appear with several input choices to make via dropdown menus and searchBox input to select the origin and destination of your flight. 
-Click ‘Ok’ after selecting all inputs. 
-A window titled ‘Team51.io’ should appear displaying the information that was input along with a delay prediction confidence. 
-The interactive map will automatically be saved to the ‘CODE’ folder as ‘this_map.html’ and also opened in your browser. 
-In case the map was not shown automatically, you can open the ‘this_map.html’ manually to check the delay probably from your selected origin airport or to your selected destination airport.
