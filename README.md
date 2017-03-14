# 3-day El Nino data challenge
Goal was to predict the temperature in the Pacific El Nino zone on a 6-month ahead basis.

## File description
- The notebook named "el\_nino..." sets up the problem and gives a piece of context
- The notebook named "Antoine..." presents different steps taken from time 0, what did not work, what looked promising but did not work, and what worked the best
- ts\_feature\_extractor.py and regressor.py enable to score top 3 on the public leaderboard
- Models can be easily trained, just running user\_test\_submission.py

## Main lessons
What worked best to predict temperatures at time t+6:
- Feature engineering:
	- Considering worldwide temperatures at time t
	- Considering average temperature in the El Nino zone at time t - 6 (standing for the same month to predict, but one year before)
- Prediction step was an average of:
	- XGBoost in order to extract most relevant geographic zones, which can be assimilated to feature selection. Best parameters were estimated using CV (see the notebook for more details)
	- Linear regression with Lasso penalization

Below is a Figure that shows how much a geographic zone leverages the temperature in the El-Niño zone:
![alt tag](https://github.com/antisrdy/el_nino/blob/master/images/temp_influence.png)

## Requirements
- XGBoost: http://xgboost.readthedocs.io/en/latest/build.html
- Some utilities. If you're using Anaconda: `conda install xarray dask netCDF4 bottleneck basemap`

## Usage
Open a terminal
~~~~
git clone https://github.com/antisrdy/el_nino
cd el_nino
python user_test_submission.py
~~~~
