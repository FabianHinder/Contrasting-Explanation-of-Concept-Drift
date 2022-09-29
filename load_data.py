# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.datasets import load_digits
from tensorflow import keras
import random
from skmultiflow.data.stagger_generator import STAGGERGenerator


mnist_cache = []
def load_mnist():
    if len(mnist_cache) == 0:
        mnist_cache.append(fetch_openml('mnist_784', version=1))
    mnist = mnist_cache[0]

    def t(X,p):
        return np.hstack((X, (np.random.random(size=X.shape[0]) < p).reshape(-1,1)))

    X = np.vstack( (t(mnist.data[mnist.target == '1'],0),t(mnist.data[mnist.target == '3'],0),t(mnist.data[mnist.target == '4'],0.5),t(mnist.data[mnist.target == '7'],1),t(mnist.data[mnist.target == '8'],1)) )
    X = X[np.random.choice(range(X.shape[0]),size=5*250)]
    return X[X[:,-1]==0][:,:-1],X[X[:,-1]==1][:,:-1]

def load_electro():
    X,_ = read_data_electricity_market()
    X = X[:,2:]
    return X[np.random.choice(range(0,17423),size=10000)],X[np.random.choice(range(17423,35000),size=10000)]    

def load_sklearn_digits():
    X_digits, y_digits = load_digits(return_X_y=True)

    def t(X,p):
        return np.hstack((X, (np.random.random(size=X.shape[0]) < p).reshape(-1,1)))

    X = np.vstack( (t(X_digits[y_digits== 1],0),t(X_digits[y_digits == 3],0),t(X_digits[y_digits == 4],0.5),t(X_digits[y_digits == 7],1),t(X_digits[y_digits == 8],1)) )
    #print(X[:,-1])
    X = X[np.random.choice(range(X.shape[0]),size=5*250)]
    return X[X[:,-1]==0][:,:-1],X[X[:,-1]==1][:,:-1]  

def create_MNIST_stream(n=500, pre_drift=[1,3], post_drift=[7,8], both=[4]):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    digits, dig_lab = np.concatenate([x_train, x_test], axis=0), np.concatenate([y_train, y_test], axis=0).astype(int)
    data = np.empty(shape=(2*n,digits.shape[1],digits.shape[2]))

    pre_drift, post_drift, both = [1,3],[7,8],[4]
    classes = np.empty(2*n,dtype=int)
    for i,j in enumerate(np.hstack( (np.random.choice( pre_drift+both, size=n, replace=True ),np.random.choice( post_drift+both, size=n, replace=True )) )):
        data[i,:] = digits[np.random.choice(np.where(dig_lab == j)[0])]
        classes[i] = j
    y = np.array(n*[0]+n*[1])
    
    return data,y,classes

# Load Activitiy Recognition data set
def load_har_data(data_path="data/HAR/"):
    subjects = []
    for i in range(1, 30):
        if i < 10:
            subjects.append(f"0{i}")
        else:
            subjects.append(f"{i}")

    X = []; y = []
    for s in subjects:
        data = np.load(os.path.join(data_path, f"{s}-segmented.npz"))
        X_, y_ = data["X"], data["y"]
        X_ = create_features(X_)
        X.append(X_);y.append(y_)
    X = np.concatenate(X)
    y = np.concatenate(y)
    print(X.shape, y.shape)

    # Select based on label (activity)
    idx1 = y == 1  # WALKING_UPSTAIRS
    idx2 = y == 2  # WALKING_DOWNSTAIRS

    X_upstairs = X[idx1, :]
    X_downstairs = X[idx2, :]

    return X_upstairs, X_downstairs

# SEA
def sea_fd(n_batches, n_samples_per_batch=100, n_features=3, x_range=(0.,10.), threshold=7):
    X = []
    y = []
    important_features = []

    for _ in range(n_batches):
        # Determine important features
        features = list(range(n_features))
        random.shuffle(features)
        relevant_features = features[:2]

        # Sample
        data = (x_range[1] - x_range[0]) * np.random.random_sample((n_samples_per_batch, n_features)) + x_range[0]

        # Compute labels
        labels = np.sum(data[:,relevant_features], axis=1) <= threshold
        labels = labels.astype(np.int)

        # Store batch
        X.append(data)
        y.append(labels)
        important_features.append(relevant_features)

    return X, y, important_features


# STAGGER
def stagger(n_batches, n_samples_per_batch=100):
    X = []
    y = []
    classifiers_ids = []
    
    for _ in range(n_batches):
        # Choose underlying classification
        h_id = random.choice([0, 1, 2])
        classifiers_ids.append(h_id)

        # Sample batch
        sg = STAGGERGenerator(classification_function=h_id, balance_classes=True)
        data, labels = sg.next_sample(n_samples_per_batch)

        # Store batch
        X.append(data)
        y.append(labels)


    return X, y, classifiers_ids



def read_data_electricity_market(foldername="data/"):
    df = pd.read_csv(foldername + "elecNormNew.csv")
    data = df.values
    X, y = data[:, 1:-1], data[:, -1]

    # Set x,y as numeric
    X = X.astype(float)
    label = ["UP", "DOWN"]
    le = LabelEncoder()
    le.fit(label)
    y = le.transform(y)

    return X, y

def read_data_weather(foldername="data/weather/"):
    df_labels = pd.read_csv(foldername + "NEweather_class.csv")
    y = df_labels.values.flatten()

    df_data = pd.read_csv(foldername + "NEweather_data.csv")
    X = df_data.values

    return X, y 


def read_data_forest_cover_type(foldername="data/"):
    df = pd.read_csv(foldername + "forestCoverType.csv")
    data = df.values
    X, y = data[:, 1:-1], data[:, -1]

    return X, y


def create_features(X):
    phi = []
    row_count = X.shape[0]

    def compute_features(X):
        features = []
        col_count = X.shape[1]

        for i in range(0, col_count):
            col = X[:, i]

            features.append(np.mean(col))
            features.append(np.var(col))
            features.append(np.median(col))

        return features

    for i in range(0, row_count):
        # Compute features
        features = compute_features(X[i])
        phi.append(features)


    phi = np.array(phi).astype(np.float)

    return phi
