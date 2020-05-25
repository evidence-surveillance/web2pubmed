"""
Authors: Eliza Harrison, Didi Surian, Paige Martin

This module contains all functions required for the feature representation and dimensionality reduction
for PubMed articles and linked webpages.

"""

from dataset_preprocessing import make_directory, pickle_dump, pickle_load, json_dump, json_load

import pandas as pd
import scipy

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def vocab_generation(pm_corpus,
                     web_corpus,
                     full_corpus,
                     **params,
                     ):
    """
    Uses CountVectorizer to generate a shared vocabulary that includes only
    those feature that are present in both the web and PubMed datasets.

    Parameters
    ----------
    pm_corpus : dataframe
        The final PubMed corpus
    web_corpus : dataframe
        The final web corpus
    full_corpus : dataframe
        The full set of web and Pubmed documents
    params : dict
        Dictionary containing the hyperparameters that define the experimental group

    Returns
    -------
    shared_vocab : The list of words present in both the PubMed and web portions of the
        dataset

    """

    shared_vocab_filename = 'shared_vocab_{}_{}.pkl'.format(params['min_dfs'],
                                                            params['max_dfs'], )

    try:
        final_shared_vocab = pd.read_pickle(shared_vocab_filename)
        print('\nShared vocabulary loaded from file ({})'.format(shared_vocab_filename))
    except FileNotFoundError:
        print('\nGenerating shared vocabulary...')
        # Generates webpage vocabulary
        vectorizer = CountVectorizer(lowercase=True)
        web_countx = vectorizer.fit(web_corpus['corpus_text'])
        web_vocab = pd.Series(vectorizer.get_feature_names())
        print('Web vocabulary size: {}'.format(len(web_vocab)))

        # Generates PubMed article vocabulary
        vectorizer = CountVectorizer(lowercase=True)
        pubmed_countx = vectorizer.fit(pm_corpus['corpus_text'])
        pubmed_vocab = pd.Series(vectorizer.get_feature_names())
        print('PubMed vocabulary size: {}'.format(len(pubmed_vocab)))

        # Generates full, combined corpus vocabulary
        vectorizer = CountVectorizer(lowercase=True,
                                     min_df=params['min_dfs'],  # Must appear in >= x articles (web or PubMed)
                                     max_df=params['max_dfs'],  # Must appear in <= yy% of articles (web or PubMed)
                                     )
        web_pm_countx = vectorizer.fit_transform(full_corpus['corpus_text'])
        web_pm_vocab = pd.Series(vectorizer.get_feature_names())
        print('\nMinimum DF: {}\nMaximum DF: {}'.format(params['min_dfs'], params['max_dfs']))
        print('Full combined corpus vocabulary size: {}'.format(len(web_pm_vocab)))

        shared_vocab = web_pm_vocab[web_pm_vocab.isin(web_vocab) & web_pm_vocab.isin(pubmed_vocab)]
        # web_only_vocab_no_threshold = web_vocab[~web_vocab.isin(pubmed_vocab)]
        # pubmed_only_vocab_no_threshold = pubmed_vocab[~pubmed_vocab.isin(web_vocab)]

        # Identifies and removes all 'numeric-only' words in vocabulary
        numeric_words = [word for word in set(shared_vocab) if word.isdigit()]
        final_shared_vocab = shared_vocab[~shared_vocab.isin(numeric_words)].reset_index(drop=True, )
        print('Final shared vocabulary size: {}'.format(len(final_shared_vocab)))

        # Saves NO THRESHOLD vocabulary to file for use in analyses
        final_shared_vocab.to_pickle(shared_vocab_filename, )

    return final_shared_vocab


def feature_representation(vocab,
                           full_corpus,
                           **params,
                           ):
    """
    Generates binary, term frequency and TF-IDF feature matrices using a pre-specified
    vocabulary of words common to both the web and PubMed datasets.

    Parameters
    ----------
    vocab : series
        Shared vocabulary containing only those words present in both the web and PubMed
        datasets
    full_corpus : dataframe
        The full set of web and Pubmed documents
    params : dict
        Dictionary containing the hyperparameters that define the experimental group

    """

    format_str = '{}_{}_{}'.format(params['feature_extraction'],
                                   params['min_dfs'],
                                   params['max_dfs'],
                                   )

    vectorizer_filename = '{}_vectorizer.pkl'.format(format_str)
    matrix_filename = '{}_matrix.npz'.format(format_str)

    try:
        feature_x = scipy.sparse.load_npz(matrix_filename)
        vectorizer = pickle_load(vectorizer_filename)
        print('Feature matrix loaded from file')
    except FileNotFoundError:
        print('\nBeginning {} vectorization...'.format(params['feature_extraction']))
        if params['feature_extraction'] == 'Binary':
            vectorizer = CountVectorizer(vocabulary=vocab.values,
                                         binary=True,
                                         )
        elif params['feature_extraction'] == 'TF':
            vectorizer = CountVectorizer(vocabulary=vocab.values,
                                         binary=False,
                                         )
        elif params['feature_extraction'] == 'TF-IDF':
            vectorizer = TfidfVectorizer(vocabulary=vocab.values, )
        else:
            raise ValueError('Invalid option for feature representation')

        print(vectorizer)
        feature_x = vectorizer.fit_transform(full_corpus['corpus_text'])
        print('Vectorization complete'
              + str(feature_x.shape))
        # Saves sparse matrix and vectorizer objects to files
        scipy.sparse.save_npz(matrix_filename, feature_x)
        pickle_dump(vectorizer, vectorizer_filename)

    return vectorizer, feature_x


def tsvd_function(feature_matrix, n_components):
    """
    Performs dimensionality reduction on a feature matrix using Truncated Singular
    Value Decomposition (T-SVD), reducing dimenions to n_components.

    Parameters
    ----------
    feature_matrix : sparse matrix
    Matrix on which to perform T-SVD (e.g. binary, term frequency or TF-IDF matrix)
    n_components : int
        Number of singular values to retain in final T-SVD matrix

    Returns
    -------
    tsvd_model, tsvd_x : tuple
        The T-SVD function object and resulting matrix.

    """

    tsvd_model = TruncatedSVD(n_components=n_components)
    tsvd_x = scipy.sparse.csr_matrix(tsvd_model.fit_transform(feature_matrix))

    print('Sum of explained variance ratio = ' + str(tsvd_model.explained_variance_ratio_.sum()))

    return tsvd_model, tsvd_x


def perform_dimensionality_reduction(feature_matrix,
                                     **params,
                                     ):
    """
    Performs dimensionality reduction (T-SVD) for relevant experimental conditions

    Parameters
    ----------
    feature_matrix : sparse matrix
        Binary, TF or TF-IDF weighted feature matrix
    params : dict
        Dictionary containing the hyperparameters that define the experimental group

    Returns
    -------
    tsvd_x : matrix
    TSVD output matrix, of dimensions n_components x n_documents

    """

    if params['dimensionality_reduction'] == 'Thresholds':
        print('\nDimensionality reduction via document frequency thresholds: {}, {}'.format(params['min_dfs'],
                                                                                            params['max_dfs'], ))
        return feature_matrix

    else:
        matrix_filename = '{}_{}_{}_TSVD_applied_{}.npz'.format(params['feature_extraction'],
                                                                params['min_dfs'],
                                                                params['max_dfs'],
                                                                params['tsvd_components'], )
        try:
            tsvd_x = scipy.sparse.load_npz(matrix_filename)
            print('T-SVD matrix loaded from file')

        except FileNotFoundError:
            tsvd_obj, tsvd_x = tsvd_function(feature_matrix,
                                             params['tsvd_components'],
                                             )

            scipy.sparse.save_npz(matrix_filename,
                                  tsvd_x,
                                  )
            print('T-SVD dimensionality reduction complete')

        return tsvd_x
