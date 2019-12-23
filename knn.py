import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["patch.force_edgecolor"] = True

#%matplotlib inline

df = pd.read_csv('heart.csv', index_col=0)
df.info()

df.describe()