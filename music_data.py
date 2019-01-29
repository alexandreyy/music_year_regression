'''
Created on 05/04/2015

@author: Alexandre Yukio Yamashita
'''

from unbalanced_dataset.ensemble_sampling import EasyEnsemble, BalanceCascade
from unbalanced_dataset.over_sampling import SMOTE, OverSampler
from unbalanced_dataset.under_sampling import UnderSampler, ClusterCentroids, \
    TomekLinks, CondensedNearestNeighbour

import numpy as np


class MusicData:
    '''
    Read music data.
    '''

    def __init__(self, path = ""):
        if path != "":
            data = np.loadtxt(path, delimiter = ',')
            m = len(data[:, 1:])
            self.X = np.hstack((np.ones((m, 1)), data[:, 1:]))
            self.y = data[:, 0]
            self.y.shape = (m, 1)

    def add_features(self, max_degree):
        '''
        Add polynomial features.
        '''
        n = len(self.X[0])

        for degree in range(2, max_degree + 1):
            self.X = np.hstack((self.X, np.power(self.X[:, 1:n], degree)))

    def add_combinarial_features(self, max_degree):
        '''
        Add polynomial features.
        '''
        n = len(self.X[0])

        for degree in range(2, max_degree + 1):
            self.X = np.hstack((self.X, np.power(self.X[:, 1:n], degree)))

    def balance_data_oversampling_random(self):
        '''
        Balance data randomly copying existent data.
        '''

        x = self.X
        y = self.y
        y.shape = (len(self.y))
        verbose = True

        OS = OverSampler(verbose = verbose)
        ox, oy = OS.fit_transform(x, y)

        self.X = ox
        self.y = oy
        self.y.shape = (len(self.y), 1)

    def balance_data_oversampling_smote_borderline1(self):
        '''
        Balance data using SMOTE bordeline 1.
        '''
        x = self.X
        y = self.y
        y.shape = (len(self.y))
        verbose = True

        sm = SMOTE(kind = 'borderline1', verbose = verbose)
        svmx, svmy = sm.fit_transform(x, y)

        self.X = svmx
        self.y = svmy
        self.y.shape = (len(self.y), 1)

    def balance_data_oversampling_smote_borderline2(self):
        '''
        Balance data using SMOTE bordeline 2.
        '''
        x = self.X
        y = self.y
        y.shape = (len(self.y))
        verbose = True

        sm = SMOTE(kind = 'borderline2', verbose = verbose)
        svmx, svmy = sm.fit_transform(x, y)

        self.X = svmx
        self.y = svmy
        self.y.shape = (len(self.y), 1)

    def balance_data_oversampling_smote_regular(self):
        '''
        Balance data using SMOTE regular.
        '''
        x = self.X
        y = self.y
        y.shape = (len(self.y))
        verbose = True

        sm = SMOTE(kind = 'regular', verbose = verbose)
        svmx, svmy = sm.fit_transform(x, y)

        self.X = svmx
        self.y = svmy
        self.y.shape = (len(self.y), 1)

    def balance_data_oversampling_smote_svm(self):
        '''
        Balance data using SMOTE SVM.
        '''
        x = self.X
        y = self.y
        y.shape = (len(self.y))
        verbose = True

        svm_args = {'class_weight': 'auto'}
        sm = SMOTE(kind = 'svm', verbose = verbose, **svm_args)
        svmx, svmy = sm.fit_transform(x, y)

        self.X = svmx
        self.y = svmy
        self.y.shape = (len(self.y), 1)

    def balance_data_undersampling_random(self):
        '''
        Balance data randomly removing existent data.
        '''
        x = self.X
        y = self.y
        y.shape = (len(self.y))
        verbose = True

        US = UnderSampler(verbose = verbose, ratio = 2)
        usx, usy = US.fit_transform(x, y)

        self.X = usx
        self.y = usy
        self.y.shape = (len(self.y), 1)

    def balance_data_undersampling_cluster_centroids(self):
        '''
        Balance data clustering centroids.
        '''
        x = self.X
        y = self.y
        y.shape = (len(self.y))
        verbose = True

        CC = ClusterCentroids(verbose = verbose)
        ccx, ccy = CC.fit_transform(x, y)

        self.X = ccx
        self.y = ccy
        self.y.shape = (len(self.y), 1)

    def balance_data_undersampling_tomek_links(self):
        '''
        Balance data clustering centroids.
        '''
        x = self.X
        y = self.y
        y.shape = (len(self.y))
        verbose = True

        TL = TomekLinks(verbose = verbose)
        tlx, tly = TL.fit_transform(x, y)

        self.X = tlx
        self.y = tly
        self.y.shape = (len(self.y), 1)

    def balance_data_ensemblesampling_condensed_nearest_neighbour(self):
        '''
        Balance data using condensed nearest neighbour.
        '''
        x = self.X
        y = self.y
        y.shape = (len(self.y))
        verbose = True

        CNN = CondensedNearestNeighbour(verbose = verbose)
        cnnx, cnny = CNN.fit_transform(x, y)

        self.X = cnnx
        self.y = cnny
        self.y.shape = (len(self.y), 1)

    def balance_data_undersampling_easy_ensemble(self):
        '''
        Balance data using easy ensemble.
        '''
        x = self.X
        y = self.y
        y.shape = (len(self.y))
        verbose = True

        EE = EasyEnsemble(verbose = verbose)
        eex, eey = EE.fit_transform(x, y)

        self.X = eex
        self.y = np.array(eey)
        self.y.shape = (len(self.y), 1)

    def balance_data_ensemblesampling_balance_cascade(self):
        '''
        Balance data using balance cascade.
        '''
        x = self.X
        y = self.y
        y.shape = (len(self.y))
        verbose = True

        BS = BalanceCascade(verbose = verbose)
        bsx, bsy = BS.fit_transform(x, y)

        self.X = bsx
        self.y = bsy
        self.y.shape = (len(self.y), 1)


    def balance_data_undersampling_custom(self, threshold = 1000, ratio = 10):
        '''
        Balance data using balance cascade.
        '''
        copy_y = np.array(self.y)
        copy_y.shape = (len(self.y))
        count = np.bincount(copy_y.astype(int))
        ii = np.nonzero(count)[0]
        count = zip(ii, count[ii])
        y = np.array([])
        y.shape = (0, 1)
        X = np.array([])
        X.shape = (0, len(self.X[0]))

        for index in range(len(count)):
            year = count[index][0]
            total_year = count[index][1]

            if total_year < threshold:
                X = np.vstack((X, self.X[np.where(copy_y == year)]))
                y = np.vstack((y, self.y[np.where(copy_y == year)]))
            else:
                X = np.vstack((X, self.X[0:threshold]))
                y = np.vstack((y, self.y[0:threshold]))

        self.X = X
        self.y = y
