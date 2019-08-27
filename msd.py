# Data manipulation
import pandas as pd
import multiprocessing as mp
from sklearn.model_selection import train_test_split

# a comment
import numpy as np
import matplotlib.pyplot as plt
from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from IPython.display import display
from sklearn import metrics


def generate_data(numb_samples):
    # create some synthetic data
    from sklearn.datasets.samples_generator import make_blobs
    # generate 2 overlapping classification dataset
    X, dep_var = make_blobs(n_samples=numb_samples, centers=[(1, 1), (1, 1)], cluster_std=3, n_features=6)
    # scatter plot, dots colored by class value
    df = pd.DataFrame(dict(x=X[:, 0], y=X[:, 1], x1=X[:, 1], x2=X[:, 1], x3=X[:, 1], x4=X[:, 1], label=dep_var))
    # ax1 = df.plot.scatter(x='x',
    #                       y='y',
    #                       c='label',
    #                       colormap='viridis')
    # plt.show()
    return df


def run_tests(trn,trn_y,tst,numb_rounds,lck,alldata):
    rets = []
    for _ in range(numb_rounds):
        # create a random forest object
        m_rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, oob_score=True, max_features='auto', min_samples_leaf=10,
                                      verbose=False);

        # train the random forest
        # the _= suppresses the output
        m_rf.fit(trn, trn_y)

        all = m_rf.predict_proba(tst)[:, 0]
        avg = all.sum() / len(all)

        rets.append(avg)
    print(avg)
    lck.acquire()
    alldata.append(avg)
    print(alldata)
    lck.release()

if __name__=="__main__":
    numb_samples = 100

    df = generate_data(numb_samples)

    #get train tst splits and place in ns namespace
    trn, tst = train_test_split(df, test_size=0.1)
    trn_y = trn['label'].copy()
    tst_y = tst['label'].copy()
    trn_y.astype('int64');
    trn_y.astype('int64');
    trn.drop('label', axis=1, inplace=True);
    tst.drop('label', axis=1, inplace=True);

    numb_rounds = 10

    all_data = []
    lck = mp.Lock() #for locking list where results wind up
    for _ in range(mp.cpu_count()-1):
        mp.Process(target=run_tests, args=(trn.copy(),trn_y.copy(), tst.copy() ,numb_rounds,lck,all_data)).start()

    # #launch 1 process per core
    # p = mp.Pool(processes=mp.cpu_count())
    # all_data = p.starmap(run_tests, [(ns, numb_rounds), (ns, numb_rounds), (ns, numb_rounds)])
    print("the data is" + str(all_data))
    pass
