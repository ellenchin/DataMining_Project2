import numpy as np 
import pandas as pd 
import os 
from sklearn import tree 
from sklearn import preprocessing 
from IPython.display import Image 

mypath = 'C:\\Users\\ellen\\Desktop'
os.chdir(mypath)

train = pd.read_csv("106.csv")
features = ["time","water","age"]
trainer=pd.DataFrame([train["time"],
						  train["water"],
						  train["age"]
							]).T
tree_model = tree.DecisionTreeClassifier(max_depth = 3)

tree_model.fit(X = trainer,y = train["survive"])

tree_model.score(X = trainer, y = train["survive"])

with open ("tree3.dot", 'w') as f:
	f = tree.export_graphviz(tree_model, feature_names=features , out_file = f)
