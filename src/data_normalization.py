"""
DATA NORMALIZATION
"""
import pandas as pd
import numpy as np
import util

df = util.create_df()
""" Overview
1) Simple Feature Scaling
    xNew = xOld/xMax
2) Min-Max
    xNew = (xOld-xMin)/(xMax-xMin)
3) Z-score
    xNew = (xOld-m)/sd
    m: Average -> mean()
    sd: Standard Deviation -> std()
"""
# Simple Feature Scaling
df['length'] = df['length'] / df['length'].max()

# Min-Max
df['length'] = (df['length'] - df['length'].min()) / \
    (df['length'].max()-df['length'].min())

# Z-score
df['length'] = (df['length'] - df['length'].mean())/df['length'].std()
