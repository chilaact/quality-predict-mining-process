# Use seaborn for pairplot
!pip install -q seaborn

# Use some functions from tensorflow_docs
!pip install -q git+https://github.com/tensorflow/docs

# Importing some useful modules

from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from google.colab import drive
drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
#Importing TensorFlow 2.x

# %tensorflow_version 2.x

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

#Linking Drive to Colab
from google.colab import drive
drive.mount('/content/drive/')

#Load the Data
data = pd.read_csv("/content/drive/My Drive/Quality_Prediction/MiningProcess_Flotation_Plant_Database.csv",
                    decimal=",",parse_dates=["date"],infer_datetime_format=True).drop_duplicates()

data.head()

data.isna().sum()



plt.figure(figsize=(20, 15))
correlation = sns.heatmap(data.corr(), cmap='YlGnBu', annot=True)

