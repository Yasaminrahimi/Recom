import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import csv
from sklearn import preprocessing
import warnings
warnings.filterwarnings("ignore")

trainset = pd.read_csv("/Users/yada/Downloads/IMDBMovieData.csv", encoding='latin-1')
X = trainset.drop(['Description', 'Runtime'], axis=1)
X.Revenue = X.Revenue.fillna(X.Revenue.mean())
X.Metascore= X.Metascore.fillna(X.Revenue.min())
features = ['Genre','Actors','Director']
for f in features:
    X_dummy = X[f].str.get_dummies(',').add_prefix(f + '.')
    X = X.drop([f], axis = 1)
    X = pd.concat((X, X_dummy), axis = 1)
    
np_df = X.as_matrix()

X = X.drop(['Title', 'ID', 'Votes', 'Year', 'Revenue','Metascore', 'Rating'], axis=1)
#X.encode('utf-8')
y = list(X.columns.values)

"""with open('/Users/yada/Downloads/testing.csv', 'w', encoding="ISO-8859-1") as test:
       write = csv.writer(test, delimiter = ",")
       for i in range(3030):
           write.writerow([y[i]])"""

test = pd.read_csv("/Users/yada/Downloads/testing.csv")
T = test.drop(['Content'], axis=1)
T = T['Vote'].fillna(0)
vote = T.values
vec = np.ones((1004,3026), dtype=np.uint8)
vec = X.values

sim = np.ones((1004,), dtype=np.complex_)
for i in range (1,1004):
    sim[i] = np.inner(vec[i],vote.transpose())

M = np.argmax(sim)
print (M)
similar = sim.argsort()[::-1][:30]
for i in range (30):
    print (np_df[similar[i]])
 
"""vec = np.ones((1000,6000), dtype=np.uint8)
vec = X.values
#print (vec)
sim = np.ones((1000,1), dtype=np.complex_)
#print (type(sim))
for i in range (1,1000):
    sim[i][0] = np.inner(vec[i],vec[1])
M = max(sim)
print (np.argmax(sim))"""
