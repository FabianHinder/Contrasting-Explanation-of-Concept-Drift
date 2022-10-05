# Contrasting Explanation of Concept Drift [1] #

This reposetory contains an reference implementation of drift explanation toolkit described in "Contrasting Explanation of Concept Drift" [1] as well as several examples (see [examples.ipynb]). 

## Description / Abstract ##
The notion of concept drift refers to the phenomenon that the distribution, which is underlying the observed data, changes over time. As a consequence machine learning models may become inaccurate and need adjustment. While there do exist methods to detect concept drift or to adjust models in the presence of observed drift, the question of _explaining_ drift is still widely unsolved. This problem is of importance, since it enables an understanding of the most prominent drift characteristics. In this work we propose to explain concept drift by means of contrasting explanations describing characteristic changes of spatial features. We demonstrate the usefulness of the explanation in several examples.

## How to use ##
The explaining routine is started when drift is detected and explanations are required. To perform the explanation take a time window containing samples from before and after (`X`) the detected drift and anotate each sample wether it was observed before (`y[i] = 0`) or after (`y[i] = 1`) the drift. Then apply the `select_samples(X,y)` method to perform a drift localization and obtain the characteristic samples. Then train a (simple) model to learn the drift structure, e.g., by using the `simple_model` method, and use it and the characteristic samples to compute the explanation, e.g., by using the `compute_counterfactual_explanation` method.

## Examples ##
```python
import numpy as np
import matplotlib.pyplot as plt
import drift_counterfactuals as dcf
from utility import umap_picture, show_sample
from sklearn.ensemble import RandomForestClassifier

## Load and Compose Dataset
X_0 = ... # Before Drift Window
X_1 = ... # After Drift Window

X = np.vstack( (X_0,X_1) )
y = np.array( X_0.shape[0]*[0]+X_1.shape[0]*[1] )

## Perform Drift Localization and find Characteristic Samples
samps,y_s, p, n,_ = dcf.select_samples(X,y, select_from_samples=True)

## Train and test compressed Model for Drift Localization 
model = dcf.simple_model(X,y,p, background=2.5,model = RandomForestClassifier())
print("Wrong timed samples before", (model.predict(X_0)==1).mean(), "Background confusion before", (model.predict(X_0)==3).mean() )
print("Wrong timed samples after ", (model.predict(X_1)==0).mean(), "Background confusion after ", (model.predict(X_1)==3).mean() )
 
## Compute Counterfactuals
cfs = dcf.compute_counterfactual_explanation(samps, y_s, model, method="assignment", X=X)
print("Classefied CS",model.predict(samps),"Classified CF",model.predict(cfs))

##Plot Results
umap_picture(X,p,n, model, samps,cfs)
plt.show()
show_sample(samps,cfs)
plt.show()
```
Further examples can be found in [examples.ipynb]

## Dependencies ##
Primary:

* numpy
* scipy
* scikit-learn
* matplotlib

Secondary / optinal:

* ceml
* umap-learn
* jax

## How to install ##

TODO 

## References

1. F. Hinder, A. Artelt, V. Vaquet, B. Hammer and M. Verleysen. ["Contrasting Explanation of Concept Drift."](https://www.esann.org/proceedings/2022) ESANN. 2022.