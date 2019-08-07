from warnings import filterwarnings
filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage,dendrogram

df=pd.read_csv("USArrests.csv",sep=',').copy()
df.index=df.iloc[:,0]
df=df.iloc[:,1:5]
del df.index.name
print("eksik gözlem",df.isnull().sum())
print("veri yapısı",df.info())
print("betimsel istatistikler",df.describe().T)
plt.figure(figsize=(8,5))
kmeans=KMeans(n_clusters=4)
k_fit=kmeans.fit(df)
print(k_fit.cluster_centers_)
print("herbir gözlemin hangi sınıfa sahip olduğu bilsisi",k_fit.labels_)

#görselleştirme
kmeans=KMeans(n_clusters=2)
k_fit=kmeans.fit(df)
kumeler=k_fit.labels_
plt.scatter(df.iloc[:,0],df.iloc[:,1],c=kumeler,s=50,cmap="viridis")
merkezler=k_fit.cluster_centers_
plt.scatter(merkezler[:,0],merkezler[:,1],c="black",s=200,alpha=0.5)


kmeans=KMeans(n_clusters=3)
k_fit=kmeans.fit(df)
plt.rcParams['figure.figsize']=(16,9)
fig=plt.figure()
ax=Axes3D(fig)
ax.scatter(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2])


#merkezlerin grafik üzerinde işaretlenmesi
ax.scatter(df.iloc[:,0],df.iloc[:,1],df.iloc[:,2],c=kumeler)
ax.scatter(merkezler[:,0],merkezler[:,1],merkezler[:,2],
           marker='*',
           c='#050505',
           s=1000)
plt.show()

#Cluster numaralarını ve hangi eyaletin hangi cluster'a ait olduğu
pd.DataFrame({"Eyaletler":df.index, "Kumeler":kumeler})
df["kume_no"]=kumeler
df["kume_no"]=df["kume_no"]+1
print(df)
