import numpy as np

en_lat_bottom = -5
en_lat_top = 5
en_lon_left = 360 - 170
en_lon_right = 360 - 120

def get_area(tas, coords):
    return tas.loc[:,coords[0]:coords[1], coords[2]:coords[3]]
    
def get_area_mean(tas, lat_bottom, lat_top, lon_left, lon_right):
    """The array of mean temperatures in a region at all time points."""
    return tas.loc[:, lat_bottom:lat_top, lon_left:lon_right].mean(dim=('lat','lon'))

def get_enso_mean(tas):
    """The array of mean temperatures in the El Nino 3.4 region at all time points."""
    return get_area_mean(tas, en_lat_bottom, en_lat_top, en_lon_left, en_lon_right)

class FeatureExtractor(object):

    def __init__(self):
        pass
        
    def transform(self, X_ds):
        valid_range = np.arange(X_ds.n_burn_in, len(X_ds['time']))
        
        worldwide = X_ds['tas'].values
        worldwide_vectorized = worldwide.reshape(len(worldwide), -1)
        X_array = worldwide_vectorized
        X_array = X_array[valid_range]
        
        enso = get_enso_mean(X_ds['tas'])
        enso_rolled_last_year = np.roll(enso, 12 - X_ds.n_lookahead)[valid_range].reshape((-1, 1))
        
        X_array = np.concatenate([X_array,
                                  enso_rolled_last_year], axis=1)
        return X_array