import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import numpy as np

from matplotlib import style
from sklearn import linear_model
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_diagnostic.data.features 
y = breast_cancer_wisconsin_diagnostic.data.targets 


# assign radius1 feature and diagnosis target from datatset
radius = X['radius1']
diagnosis = y['Diagnosis'].map({"M":0, "B":1})

# assign colormap
colorMap = {0: "r", 1: "b"}

# create logistic model
lgr = linear_model.LogisticRegression()

# train the model
lgr.fit(X = np.array(radius).reshape(len(radius),1),
        y = diagnosis)

# print the intercept
modelIntercept = lgr.intercept_
print(modelIntercept)

# print the coefficients
modelCoefficients = lgr.coef_
print(modelCoefficients)

# define sigmoid function
def sigmoid(x):
    return (1 / (1 +
                 np.exp(-(modelIntercept[0] +
                          modelCoefficients[0][0] * x)
                        )                
                )
            )

# Plot the sigmoid curve
x1 = np.arange(0, 30, 0.01)
y1 = [sigmoid(n) for n in x1]

style.use("ggplot")

plt.scatter(radius, diagnosis,
            facecolors='none',
            edgecolors=diagnosis.map(colorMap)
            )

plt.plot(x1,y1)
plt.title("Logistic Regression Prediction Using Radius")
plt.xlabel("Radius")
plt.ylabel("Probability")

# Plot threshold line (0.5)
plt.plot([-2,32],[0.5,0.5],
         color='grey',
         ls='--')

# Plot the legend
r = patches.Patch(color='red', label='Malignant')
b = patches.Patch(color='b', label='Benign')
t = patches.Patch(color='grey', linestyle="--", label="Threshold")

plt.legend(handles=[r,b,t], loc=1)
plt.show()


# Predictions when tumor radius (tr) is 18 and 12
for r in [18,12]:
    parg = np.array(r).reshape(-1,1)
    print('--- Model prediction when radius =', r)
    print('Prediction Probabilities:', lgr.predict_proba(parg))
    print('The tumor is probably',
          'malignant' if lgr.predict(parg)[0] == 0 else 'benign')
    print('')


