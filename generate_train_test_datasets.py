import config
from dataset_preprocessing import make_directory

import os
import numpy as np
import pandas as pd
import pickle
import json
from collections import OrderedDict


def assign_train_test(train_portion=config.train_portion):
    """
    Assigns known URL to PMID links to either the train or test datasets.

    Parameters
    ----------
    train_portion : float
        Proportion of known links assigned to the training dataset.
        Defaults to the training portion specified in config.py.

    Returns
    -------
    known_links : dataframe
        The final set of known links, including train/test dataset assignments

    """

    try:
        known_links = pd.read_pickle(config.final_known_links_file + '.pkl')
        if not known_links.columns.contains('train_test'):
            # Generates training/test sets
            # NOTE: PubMed training set contains only those articles with known links webpages in the training set
            train_size = int(round(len(known_links) * train_portion, 0))
            test_size = len(known_links) - train_size
            corpora_links_train = known_links.sample(n=train_size)
            known_links['train_test'] = np.where(known_links.index.isin(corpora_links_train.index),
                                                 'Train',
                                                 'Test', )
            known_links.to_pickle(config.final_known_links_file, )
        return known_links
    except FileNotFoundError as e:
        print(e)


def gen_train_test_corpus():
    """
    Splits PubMed and web links corpus into train and test datasets.

    Returns
    -------
    pm_corpus, web_corpus, full_corpus, known_links : tuple
        pm_corpus: dataframe
             The final PubMed corpus loaded from file
        web_corpus: dataframe
            The final web corpus loaded from file
        full_corpus: dataframe
            The full corpus, containing web and PubMed records organised in
            the following order: web train, web test, PubMed train, PubMed test
        known_links: dataframe
            The final set of known links loaded from file

    """
    os.chdir(config.full_datasets_path)  # Directory containing full PubMed and webpage datasets
    pm_corpus = pd.read_pickle(config.final_pm_corpus_file + '.pkl')
    web_corpus = pd.read_pickle(config.final_web_corpus_file + '.pkl')
    known_links = assign_train_test()

    make_directory(config.train_test_datasets)
    os.chdir(config.train_test_datasets)  # Directory for datasets split according to train/test portions
    known_links_train = known_links.loc[known_links['train_test'] == 'Train']
    known_links_test = known_links.loc[known_links['train_test'] == 'Test']
    pm_corpus['train_test'] = np.where(pm_corpus.index.isin(known_links_train['pmid']),
                                       'Train',
                                       'Test', )
    web_corpus['train_test'] = np.where(web_corpus.index.isin(known_links_train['web_id']),
                                        'Train',
                                        'Test', )

    # Specific order to allow for later separation into individual web/PubMed training and test datasets
    pm_corpus.sort_values(['train_test', 'pmid'],
                          ascending=[False, True],
                          inplace=True, )
    pm_corpus_text = pm_corpus[['corpus_text', 'train_test']]
    web_corpus.sort_values(['train_test', 'web_id'],
                           ascending=[False, True],
                           inplace=True, )
    web_corpus_text = web_corpus[['corpus_text', 'train_test']]

    try:
        full_corpus = pd.read_pickle('full_web_pm_corpus_text.pkl')
        print('\nCorpora loaded from file.')

    except FileNotFoundError as e:
        all_train_datasets = {
            'known_links_train': known_links_train,
            'web_corpus_train': web_corpus.loc[web_corpus['train_test'] == 'Train'],
            'pm_corpus_train': pm_corpus.loc[pm_corpus['train_test'] == 'Train'],
        }
        pickle.dump(all_train_datasets,
                    open('all_train_datasets.pkl', 'wb'), )

        all_test_datasets = {
            'known_links_test': known_links_test,
            'web_corpus_test': web_corpus.loc[web_corpus['train_test'] == 'Test'],
            'pm_corpus_test': pm_corpus.loc[pm_corpus['train_test'] == 'Test'],
        }
        pickle.dump(all_test_datasets,
                    open('all_test_datasets.pkl', 'wb'), )

        # Generates full combined corpus appending all sample web articles to all PubMed articles
        pickle.dump({'full_web': web_corpus_text,
                     'full_pm': pm_corpus_text,
                     },
                    open('full_web_pm_matrix_order.pkl', 'wb'), )
        full_corpus = web_corpus_text.append(pm_corpus_text)
        full_corpus.to_pickle('full_web_pm_corpus_text.pkl', )

        corpus_lengths = {
            'web_corpus_train': len(web_corpus.loc[web_corpus['train_test'] == 'Train']),
            'web_corpus_test': len(web_corpus.loc[web_corpus['train_test'] == 'Test']),
            'pm_corpus_train': len(pm_corpus.loc[pm_corpus['train_test'] == 'Train']),
            'pm_corpus_test': len(pm_corpus.loc[pm_corpus['train_test'] == 'Test']),
            'full_web': len(web_corpus),
            'full_pm': len(pm_corpus),
        }
        json.dump(corpus_lengths,
                  open('test_train_corpus_lengths.json', 'w'),
                  indent=4, )

        print('\n{} web documents + {} PubMed documents in training set'.format(corpus_lengths['web_corpus_train'],
                                                                                corpus_lengths['pm_corpus_train']))
        print('{} web documents + {} PubMed documents in test set'.format(corpus_lengths['web_corpus_test'],
                                                                          corpus_lengths['pm_corpus_test']))
        print('There are {} known links in the training set and {} known links in the test set'.format(
            len(known_links_train),
            len(known_links_test)), )

    return pm_corpus, web_corpus, full_corpus, known_links


def load_datasets():
    """
    Loads web and PubMed datasets from files into dictionary for later vectorization.

    Returns
    -------
    all_data : dict
        Dictionary containing the following datasets:
            corpus_lengths : dict
                Lengths of training/test corpora which can be used to split feature matrices
            full_web_pm_corpus : dataframe
                Combined web and PubMed dataset (all documents)
            corpora_links_train : dataframe
                Known URL to PMID links in the training dataset
            web_corpus_train : dataframe
                Webpages in the training dataset
            pm_corpus_train : dataframe
                PubMed articles in the training dataset
            corpora_links_test : dataframe
                Known URL to PMID links in the testing dataset
            web_corpus_test : dataframe
                Webpages in the testing dataset
            pm_corpus_test : dataframe
                PubMed articles in the testing dataset
            full_web : dataframe
                Data for all webpages in the final corpus
            full_pm : dataframe
                Data for all PubMed articles in the final corpus

    """

    cwd = os.getcwd()
    os.chdir(config.train_test_datasets)   # Directory for datasets split according to train/test portions
    full_train_corpus = pd.read_pickle('all_train_datasets.pkl')
    full_test_corpus = pd.read_pickle('all_test_datasets.pkl')
    full_web_pm_matrix_order = pickle.load(open('full_web_pm_matrix_order.pkl',
                                                'rb', ))

    all_data = {
        'corpus_lengths': json.load(open('test_train_corpus_lengths.json',
                                         'r', ), object_pairs_hook=OrderedDict),
        'full_web_pm_corpus': pd.read_pickle('full_web_pm_corpus_text.pkl'),
    }
    all_data.update(full_train_corpus)
    all_data.update(full_test_corpus)
    all_data.update(full_web_pm_matrix_order)

    print('Datasets in dictionary: \n{}'.format([key for key in all_data.keys()]))

    # Returns to original working directory
    os.chdir(cwd)

    return all_data


if __name__ == '__main__':
    # GENERATE TRAIN / TEST DATASETS #
    # Generates train/test datasets according to pre-defined tags
    final_pm_corpus, final_web_corpus, combined_corpus, final_corpora_links = gen_train_test_corpus()
