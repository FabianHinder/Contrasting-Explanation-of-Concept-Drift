import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import binom
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import euclidean_distances as dist
from scipy.optimize import linear_sum_assignment as assign
from utility import find_closest_sample
from ceml.sklearn import generate_counterfactual


def select_samples(X,y,n_samples=10,k=10,p_thr=0.05, select_from_samples=False):
    model = RandomForestClassifier(criterion='gini',min_samples_leaf=k+1)
    res = (k*model.fit(X,y).predict_proba(X)[np.vstack( (y==0,y==1) ).T]-1)/(k-1)
    p = (1-binom.cdf(res*(k-1), k-1, y.mean()))
    sel = p < p_thr

    n01 = min(n_samples,sel.shape[0]/2)
    n0,n1 = int(n01*(1-y[sel].mean())),int(n01*y[sel].mean())
    original_samples = np.vstack( (
        MiniBatchKMeans(n_clusters=n0).fit(X[sel][y[sel] == 0]).cluster_centers_, 
        MiniBatchKMeans(n_clusters=n1).fit(X[sel][y[sel] == 1]).cluster_centers_) )
    if select_from_samples:
        original_samples = find_closest_sample(X, original_samples)
    original_labels = np.array(n0*[0]+n1*[1])
    return original_samples,original_labels, p, n0, n1 

def simple_model(X,y,p,p_thr=0.05,background=0.5, model = DecisionTreeClassifier(max_leaf_nodes=25)):
    X = X.copy()
    y = y.copy()
    y[p > p_thr] = 2
    
    if background != None:
        if type(background) is float:
            background = int(background * X.shape[0])
        bkX = np.random.random( size=(background,X.shape[1]) )
        for i in range(bkX.shape[1]):
            mi,ma = X[:,i].max(),X[:,i].min()
            bkX[:,i] = bkX[:,i]*(ma-mi) + mi
        X = np.vstack( (X,bkX) )
        y = np.hstack( (y,np.array(background*[3])) )
    
    model.fit(X,y)
    
    return model

def compute_counterfactual_explanation(original_samples, original_labels, model, method="ceml", X=None, regularization="l1"):
    if method=="assignment":
        y = model.predict(X)
        
        cfs = []
        for i in [0,1]:
            identy = np.argwhere(y==1-i).reshape(-1)
            cfs.append( X[identy[assign( dist(original_samples[original_labels==i], X[identy]) )[1]]] )
        
        cfs = np.vstack(cfs) 
        return cfs
    elif method=="ceml":
        orig, cfs = [], []
        for i in range(original_samples.shape[0]):
            if original_labels[i] != 2:
                x,y_t = original_samples[i,:], original_labels[i]
                y_cf = 1-y_t
                print("True label on x: {1}, Prediction on x: {0}, Aiming for label: {2}".format(model.predict([x])[0], y_t, y_cf), end="")
                if type(regularization) == type(lambda x:x):
                    regul = regularization(x)
                else:
                    regul = regularization
                cf =  generate_counterfactual(model, x, y_target=y_cf, regularization=regul)
                print(", Found: {0}, Succsess: {1}".format(cf["y_cf"], cf["y_cf"]==y_cf))
                if cf["y_cf"]==y_cf:
                    orig.append(x)
                    cfs.append(cf["x_cf"])
        return np.array(cfs)
    else:
        raise ValueError()