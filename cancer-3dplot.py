import matplotlib.pyplot as plt
import pandas as pd

# %%
from mpl_toolkits.mplot3d import Axes3D
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_diagnostic.data.features 
y = breast_cancer_wisconsin_diagnostic.data.targets 


# %%
colorMap = {"M":"r","B":"b"}
fig = plt.figure(figsize=(18,15))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X['radius1'],
           X['texture1'],
           X['perimeter1'],
           c=y['Diagnosis'].map(colorMap)
           )

ax.set_xlabel("radius1")
ax.set_ylabel("texture1")
ax.set_zlabel("perimiter1")
plt.title("Breast Cancer Scatter Plot - Three Features")
plt.show()



# %%
