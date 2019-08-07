from warnings import filterwarnings
filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage,dendrogram

df=pd.read_csv("USArrests.csv",sep=',').copy()
df.index=df.iloc[:,0]
df=df.iloc[:,1:5]
#del df.index.name
df.head()
df=StandardScaler().fit_transform(df)
pca=PCA(n_components=2)
pca_fit=pca.fit_transform(df)
bilesen_df=pd.DataFrame(data=pca_fit,
                        columns=["birinci_bilesen","ikinci_bilesen"])
print(pca.explained_variance_ratio_)
pca=PCA().fit(df)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.show()
