from warnings import filterwarnings
filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage,dendrogram

df=pd.read_csv("USArrests.csv",sep=',').copy()
df.index=df.iloc[:,0]
df=df.iloc[:,1:5]
del df.index.name
df.head()
hc_complete=linkage(df,"complete")
hc_average=linkage(df,"average")
hc_single=linkage(df,"single")
plt.figure(figsize=(15,10))
plt.title('Hiyerarşik kümeleme-Dendogram')
plt.xlabel('Indexler')
plt.ylabel('Uzaklık')
dendrogram(hc_complete,
           #leaf_font_size=10,
           #truncate_mode="lastp",
           #p=12,
           #show_contracted=True
           )
#plt.show()
cluster=AgglomerativeClustering(n_clusters=4,
                                affinity="euclidean",
                                linkage="ward")
cluster.fit_predict(df)
x=pd.DataFrame({"Eyaletler":df.index,"Kümeler":cluster.fit_predict(df)})[0:10]
x1=df["kume_no"]=cluster.fit_predict(df)
print(x,x1)
