from warnings import filterwarnings
filterwarnings('ignore')
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import pandas as pd

df=pd.read_csv("USArrests.csv",sep=',').copy()
df.index=df.iloc[:,0]
df=df.iloc[:,1:5]
del df.index.name
kmeans=KMeans()
visualizer=KElbowVisualizer(kmeans,k=(2,20))
visualizer.fit(df)
visualizer.poof()
kmeans=KMeans(n_clusters=4)
k_fit=kmeans.fit(df)
kumeler=k_fit.labels_
pd.DataFrame({"Eyaletler":df.index, "Kumeler":kumeler})

