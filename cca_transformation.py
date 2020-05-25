"""
Authors: Didi Surian, Eliza Harrison, Paige Martin

This module contains all functions required for the transformation of features using
CCA, ranking documents using cosine similarity and calculation of final evaluation
metrics.

"""
from __future__ import division
from dataset_preprocessing import pickle_dump, pickle_load

import pandas as pd
import pickle

import scipy
import scipy.sparse

from sklearn.cross_decomposition import CCA


def split_matrix(feature_matrix, part_1_n):
    """
    Splits a single sparse matrix along the row axis into two parts
    of size part_1_n and len(feature_matrix) - part_1_n

    Parameters
    ----------
    feature_matrix : sparse matrix
        Full matrix to be broken into parts of n-size.
    part_1_n : int
        Size of first part (row at which matrix is split)

    Returns
    -------
    part_1_x, part_2_x : tuple
        The two parts of the split matrix

    """

    # Selects first part_1_n rows in the feature matrix
    part_1_x = scipy.sparse.csr_matrix(feature_matrix[:part_1_n])

    # Selects the remaining rows in feature matrix
    part_2_x = scipy.sparse.csr_matrix(feature_matrix[part_1_n:])

    return part_1_x, part_2_x


def gen_train_test_matrix(reduced_feature_matrix,
                          corpus_lengths,
                          **params):
    """
    Splits the full feature matrix (post feature representation or TSVD) into
    train + portions. Uses slice indices previously saved to file during the generation
    of the train + test splits.

    Parameters
    ----------
    corpus_lengths : dict
        Dictionary containing the lengths of the web_train, web_test, pm_train and pm_test
        portions of the dataset and which is used to split the feature matrix.
    reduced_feature_matrix : sparse matrix
        Feature matrix, with the number of features reduced either using threshold paramters
        or via TSVD
    params : dict
        Dictionary containing the hyperparameters that define the experimental group

    Returns
    -------
    all_train_test_data : list
        List containing the web_train, web_test, pm_train and pm_test portions of the
        feature matrix (in that order)

    """

    format_str = '{}_{}_{}_matrix'.format(params['feature_extraction'],
                                          params['min_dfs'],
                                          params['max_dfs'],
                                          )
    if params['dimensionality_reduction'] == 'TSVD':
        format_str = '{}_{}'.format(format_str,
                                    params['tsvd_components'], )

    train_test_filenames = ('_web_train.npz',
                            '_web_test.npz',
                            '_pm_train.npz',
                            '_pm_test.npz',)
    all_files = [format_str + sfx for sfx in train_test_filenames]

    try:
        all_train_test_data = [scipy.sparse.load_npz(file) for file in all_files]
    except FileNotFoundError:
        # TRAIN & TEST DATASETS #
        print('\nSplitting TRAIN and TEST portions of feature matrix...')

        # Splits feature matrix into parts containing web and PubMed vectors
        web_feature_matrix, pm_feature_matrix = split_matrix(reduced_feature_matrix,
                                                             corpus_lengths['full_web'],
                                                             )
        # Splits web matrix in to train and test datasets
        web_train_matrix, web_test_matrix = split_matrix(web_feature_matrix,
                                                         corpus_lengths['web_corpus_train'],
                                                         )
        # Splits PubMed matrix in to train and test datasets
        pm_train_matrix, pm_test_matrix = split_matrix(pm_feature_matrix,
                                                       corpus_lengths['pm_corpus_train'],
                                                       )

        all_train_test_data = [web_train_matrix,
                               web_test_matrix,
                               pm_train_matrix,
                               pm_test_matrix, ]

        for i, x in enumerate(all_train_test_data):
            scipy.sparse.save_npz(all_files[i],
                                  x, )

    return all_train_test_data


def perform_cca(reduced_feature_matrix,
                corpus_lengths,
                **params):
    """
    Where required, transforms the feature matrix using CCA.

    Parameters
    ----------
    reduced_feature_matrix : sparse matrix
       Feature matrix following dimensionality reduction e.g. via T-SVD
    corpus_lengths : dict
        Dictionary containing the lengths of the web_train, web_test, pm_train and pm_test
        portions of the dataset and which is used to split the feature matrix.
    params : dict
        Dictionary containing the hyperparameters that define the experimental group

    Returns
    -------
        web_test_matrix_cca, pm_test_matrix_cca : tuple
        The transformed feature matrices following CCA. For experimental groups not requiring
        CCA, returns the matrices passed argument to the function.

    """

    # SPLIT FEATURE MATRIX INTO TRAIN + TEST #
    train_test_keys = ('web_train_matrix',
                       'web_test_matrix',
                       'pm_train_matrix',
                       'pm_test_matrix',)
    train_test_values = gen_train_test_matrix(reduced_feature_matrix,
                                              corpus_lengths,
                                              **params, )
    train_test_dict = dict(zip(train_test_keys,
                               train_test_values), )
    print(train_test_dict)

    # CANONICAL COVARIATE ANALYSIS #
    if params['cca'] and \
            params['tsvd_components'] > params['cca_components'] and \
            params['tsvd_components'] != 1600 and \
            params['cca_components'] not in [1600, 800]:

        X_train = train_test_dict['web_train_matrix'].toarray()
        X_test = train_test_dict['web_test_matrix'].toarray()
        Y_train = train_test_dict['pm_train_matrix'].toarray()
        Y_test = train_test_dict['pm_test_matrix'].toarray()

        print('\nBeginning CCA...')
        # Fits CCA model to TRAIN data
        cca_model = CCA(n_components=params['cca_components'],
                        max_iter=5000, )

        print('Fit...'),
        cca_model.fit(X_train, Y_train)
        '''
        print('Saving model...'),
        pickle_dump(cca_model, '{}_model.pkl'.format(params['format_str']))
        '''

        print('Transform train...')
        X_train_transform, Y_train_transform = cca_model.transform(X_train,
                                                                   Y_train, )
        web_train_matrix_cca = scipy.sparse.csr_matrix(X_train_transform)
        pm_train_matrix_cca = scipy.sparse.csr_matrix(Y_train_transform)

        '''
        print('Dump transformed train...')
        scipy.sparse.save_npz('{}_web_train.npz'.format(params['format_str']),
                              web_train_matrix_cca)
        scipy.sparse.save_npz('{}_web_train.npz'.format(params['format_str']),
                              web_train_matrix_cca)
        '''

        print('Transform test...')
        X_test_transform, Y_test_transform = cca_model.transform(X_test,
                                                                 Y_test, )
        web_test_matrix_cca = scipy.sparse.csr_matrix(X_test_transform)
        pm_test_matrix_cca = scipy.sparse.csr_matrix(Y_test_transform)
        '''
        print('Dump transformed test...'),
        scipy.sparse.save_npz('{}_web_train.npz'.format(params['format_str']),
                              web_test_matrix_cca)
        scipy.sparse.save_npz('{}_web_train.npz'.format(params['format_str']),
                              web_test_matrix_cca)
        '''
    else:
        # NO CCA REQUIRED
        print('\nNo CCA required for this experimental group')
        web_test_matrix_cca = train_test_dict['web_test_matrix']
        pm_test_matrix_cca = train_test_dict['pm_test_matrix']

    return web_test_matrix_cca, pm_test_matrix_cca
