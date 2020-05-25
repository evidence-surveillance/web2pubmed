"""
Authors: Eliza Harrison, Didi Surian, Paige Martin

This module contains all functions required for computing the similarity between PubMed and web
documents and the associated evaluation metrics.

"""

import pandas as pd
import pickle

from sklearn.metrics.pairwise import cosine_similarity


def perform_cosine_similiarity(web_test_matrix_final,
                               pm_test_matrix_final,
                               web_test_corpus,
                               pm_test_corpus,
                               known_links_test,
                               **params,
                               ):
    """
    Computes the cosine similirity between each webpage and PubMed article in the test
    dataset.

    Parameters
    ----------
    web_test_matrix_final : saprse matrix
        Matrix containing vectors corresponding to documents in the TEST portion of the
        WEBPAGE dataset
    pm_test_matrix_final : sparse_matrix
        Matrix containing vectors corresponding to documents in the TEST portion of the
        PUBMED dataset
    web_test_corpus : dataframe
        Contains the IDs and corpus_text for webpages in the TEST portion of the dataest.
        Used to generate index for dataframe containing cosine similarities.
    pm_test_corpus : dataframe
        Contains the IDs and corpus_text for PubMed articles in the TEST portion of the dataest.
        Used to generate columns for dataframe containing cosine similarities.
    known_links_test : dataframe
        Final set of known links between the web and PubMed datasets.
    params : dict
        Dictionary containing the hyperparameters that define the experimental group

    Returns
    -------
    correct_link_srs : series
        Series containing correct the ranks of the correct PubMed documents for each

    """

    correct_ranks_file = '{}_correct_link_ranks.pkl'.format(params['format_str'])

    try:
        correct_links_srs = pd.read_pickle(correct_ranks_file)
    except FileNotFoundError:

        # COSINE SIMILARITY #
        print('\nBeginning cosine similarity calculations...')

        # Cosine distance between web and PubMed vectors in tf-idf representation following transformation using CCA
        cosine_matrix = cosine_similarity(web_test_matrix_final, pm_test_matrix_final)
        print('Cosine similarity complete:'
              + str(cosine_matrix.shape))

        # Convert to dataframe
        cosine_df = pd.DataFrame(cosine_matrix,
                                 index=web_test_corpus.index,
                                 columns=pm_test_corpus.index,
                                 )

        # RANKING OF CORRECT LINKS #
        print('\nBeginning ranking...')
        correct_ranks_all = []

        for idx in known_links_test.index:
            # Saves web_id and PMID for each link to objects
            web_id = known_links_test.loc[idx, 'web_id']
            pmid = known_links_test.loc[idx, 'pmid']

            # Ranking PubMed articles based on cosine similarity
            ranks = cosine_df.loc[web_id].sort_values().rank(axis=0,
                                                             method='min',
                                                             ascending=False, )
            correct_link_rank = ranks[pmid]
            correct_ranks_all.append(correct_link_rank)

        correct_links_srs = pd.Series(correct_ranks_all,
                                      index=known_links_test['web_id'],
                                      name=params['format_str'], )
        correct_links_srs.to_pickle('{}_correct_link_ranks.pkl'.format(params['format_str']))
        print('Final ranks saved to file')

    return correct_links_srs


def measures(correct_links_srs,
             **params,
             ):
    """
    Parameters
    ----------
    correct_links_srs : series
        Series containing correct the ranks of the correct PubMed documents for each
            web document
    params : dict
        Dictionary containing the hyperparameters that define the experimental group

    Returns
    -------
    results_data : dict
        Dictionary containing all evalutation metrics (median rank, IQR 25, IQR 75,
        percentage correct, percentage in top 50, percentage in top 250)

    """

    # METRIC 1) MEDIAN RANK
    cosine_metrics = correct_links_srs.describe()

    median = cosine_metrics['50%']
    iqr_25 = cosine_metrics['25%']
    iqr_75 = cosine_metrics['75%']

    # METRIC 2) PERCENTAGE CORRECT
    cosine_correct = round(len(correct_links_srs.loc[correct_links_srs == 1]) /
                           len(correct_links_srs) * 100, 2)

    # METRIC 3) PERCENTAGE IN TOP 50
    cosine_top_50 = round((len(correct_links_srs.loc[correct_links_srs <= 50]) /
                           len(correct_links_srs)) * 100, 2)

    # METRIC 3) PERCENTAGE IN TOP 100
    cosine_top_250 = round((len(correct_links_srs.loc[correct_links_srs <= 250]) /
                            len(correct_links_srs)) * 100, 2)

    results_data = params
    results_data.update({
        'Median Rank': median,
        'IQR_25': iqr_25,
        'IQR_75': iqr_75,
        'Percentage correct': cosine_correct,
        'Percentage in Top 50': cosine_top_50,
        'Percentage in Top 250': cosine_top_250,
    })

    pickle.dump(results_data,
                open('{}_results.pkl'.format(params['format_str']),
                     'wb'),
                protocol=2, )

    print('\nFinal results saved to file')

    return results_data
