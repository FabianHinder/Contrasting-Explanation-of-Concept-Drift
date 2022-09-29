import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances as dist
from scipy.optimize import minimize
from umap import UMAP

try:
    import jax.numpy as npx
    from ceml.backend.jax.costfunctions import CostFunctionDifferentiableJax
    class MyRegularization(CostFunctionDifferentiableJax):
        def __init__(self, x_orig, mad):
            self.x_orig = x_orig
            self.mad = mad

            super(MyRegularization, self).__init__()
        
        def score_impl(self, x):
            return npx.dot(self.mad, npx.square(x - self.x_orig))
except expression as e:
    print(e)


def find_closest_sample(X,S,d=dist):
    return X[d(X,S).argmin(axis=0)]



def marker(projections, front, back, delta=None):
    mpj = projections if delta is None else projections+delta
    
    plt.scatter(mpj[:,0],mpj[:,1],s=260*5,c=back,alpha=0.75,marker="D", edgecolors=front)
    for i in range(projections.shape[0]):
        plt.plot( [projections[i,0],mpj[i,0]],[projections[i,1],mpj[i,1]], alpha=0.75, color="k")
        plt.text(mpj[i,0],mpj[i,1],str(i+1),c=front,fontsize=18*2, horizontalalignment='center', verticalalignment='center')

def umap_picture(X,p,n, model, samps,cfs):
    umap = UMAP()
    umap.fit(X)
    x = umap.transform(X)
    
    plt.figure(figsize=(20,20))
    plt.scatter(x[:,0],x[:,1],marker="o",
             c=np.log(p), 
             edgecolor=plt.get_cmap("tab10")(model.predict(X).astype(int)),
             linewidths=2)
    plt.xticks([])
    plt.yticks([])

    pr = umap.transform(np.vstack( (samps,cfs) ))
    fun = 1000
    for _ in range(5):
        res = minimize(lambda delta: 
                   (np.abs(delta)**2).mean() + 
                   (1/(dist(pr+delta.reshape(pr.shape))+1e-2+np.eye(pr.shape[0]))).max(), 
                   1e-2*np.random.normal(size=pr.shape) )
        if fun > res["fun"]:
            fun = res["fun"]
            delta = res["x"].reshape(pr.shape)

    n = samps.shape[0]
    marker(pr[:n], "k", "w", delta=delta[:n])
    marker(pr[n:], "w", "k", delta=delta[n:])

def show_images(samps, cfs, decoder=None, show_diff=False, width=2, height=2, im_cmap=plt.get_cmap("gray"), diff_cmap=plt.get_cmap("bwr")):
    if decoder is None:
        decoder = lambda x:x
    rows = 2 if not show_diff else 3
    fig, axs = plt.subplots(rows, samps.shape[0])
    fig.set_figwidth( samps.shape[0]*width )
    fig.set_figheight( rows*height )
    
    axs = axs.T
    
    axs[0,0].set_ylabel("Sample")
    axs[0,1].set_ylabel("CF")
    if show_diff:
        axs[0,2].set_ylabel("Diff.")
    
    for i,(s,c) in enumerate(zip(samps,cfs)):
        s,c = decoder(s), decoder(c)
        axs[i,0].set_title(str(i+1))
        axs[i,0].imshow(s, cmap=im_cmap)
        axs[i,1].imshow(c, cmap=im_cmap)
        if show_diff:
            axs[i,2].imshow(c-s, cmap=diff_cmap)
    for ax in axs.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

def show_sample(samps, cfs, show_diff=False, im_cmap=plt.get_cmap("rainbow"), diff_cmap=plt.get_cmap("bwr")):
    rows = 2 if not show_diff else 3
    fig, axs = plt.subplots(1, rows)
    
    axs[0].set_title("Sample")
    axs[0].set_xticks([])
    
    axs[1].set_title("CF")
    axs[1].set_xticks([])
    axs[1].set_yticks( [] )
    
    axs[0].imshow(samps, cmap=im_cmap)
    axs[1].imshow(cfs, cmap=im_cmap)
    if show_diff:
        axs[2].set_title("Diff.")
        axs[2].set_xticks([])
        axs[2].set_yticks( [] )
        axs[2].imshow((samps-cfs), cmap=diff_cmap)

def hists(X,model):
    y = model.predict(X)
    fig, axs = plt.subplots(1, np.unique(y).shape[0])
    for i,y_i in enumerate(np.unique(y)):
        axs[i].hist(np.log(p[y==y_i]+1e-132))
        axs[i].set_title(str(y_i))