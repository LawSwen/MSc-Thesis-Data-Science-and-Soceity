'''All the packages that are needed for the EDA and Modeling'''


import os
import random
import numpy as np
import tensorflow as tf

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)



from datetime import datetime   
import pandas as pd
from sktime.utils.plotting import plot_correlations
from pandas.plotting import autocorrelation_plot
from sktime.transformations.series.acf import AutoCorrelationTransformer
from sktime.transformations.series.detrend import STLTransformer
from copy import deepcopy
from copy import copy
from sktime.utils.plotting import plot_series
import warnings
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
from tsmoothie.smoother import KalmanSmoother
from statsmodels.tsa.seasonal import STL
from pandas.plotting import lag_plot
from matplotlib.lines import Line2D
from statsmodels.tsa.stattools import acovf
from statsmodels.tsa.stattools import adfuller
import keras
from sklearn.preprocessing import StandardScaler 
from os.path import join
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, GRU, Dropout, Dense, TimeDistributed, BatchNormalization)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf 
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, mean_squared_error
from math import sqrt
import logging
import optuna
import tqdm
from keras.models import Sequential
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.layers import RepeatVector
from sktime.utils.plotting import plot_series
from tensorflow.keras.layers import GRU, Dropout, BatchNormalization, RepeatVector, TimeDistributed, Dense, Concatenate
np = np
pd = pd
plt = plt
sns = sns
StandardScaler = StandardScaler
tf = tf
globals()['tf'] = tf

plt.rcParams.update({
    # Figure settings
    'figure.figsize': (12, 8),
    'figure.dpi': 100,
    'figure.autolayout': True,
    
    # Font settings
    'font.size': 12,
    'font.family': 'DejaVu Sans',
    'font.weight': 'normal',
    
    # Axes settings
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'axes.titleweight': 'bold',
    'axes.labelweight': 'normal',
    'axes.grid': True,
    'axes.edgecolor': 'black',
    'axes.linewidth': 0.8,
    
    # Grid settings
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    
    # Line settings
    'lines.linewidth': 3,
    'lines.markersize': 6,
    
    # Legend settings
    'legend.fontsize': 12,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': 'black',
    
    # Tick settings
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    
    # Savefig settings
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.transparent': False
})

__all__ = ["np", "pd", "plt", "sns", "StandardScaler", "tf"]
