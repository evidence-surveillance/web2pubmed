"""
Authors: Eliza Harrison, Didi Surian, Paige Martin

This program runs experiments to link webpages to the PubMed articles they reference
using information retrieval and document similarity methods.

"""

import config
from dataset_preprocessing import make_directory, pickle_dump, pickle_load, json_dump, json_load
from generate_train_test_datasets import load_datasets
import feature_representation_functions as ft_x
import doc_sim_rank_functions as doc_sim_rank
import cca_transformation as cca

import os
import argparse
import pickle
import pandas as pd
import json
import itertools as it
from string import ascii_uppercase
from collections import OrderedDict

# test = True  # Indicates whether to run program for subset of experimental groups


def gen_params_str(**params):
    return '_'.join([str(value) for value in params.values()])


def gen_experimental_groups(**all_params):
    """
    Generates dictionaries containing all hyperparameters for each experimental x_group.

    Parameters
    ----------
    all_params : dict
        Nested dictionary containing parameter options for Thresholds + no CCA,
        TSVD + no CCA and TSVD + CCA. Each nested dictionary contains the following keys:
            feature_extraction_methods e.g. [Binary, TF, TF-IDF]
            dimensionality_reduction_methods e.g. [TSVD]
            tsvd_components e.g. [100, 200, 400, 600, 800, 1600]
            min_dfs e.g. [1]
            max_dfs e.g. [1.0]
            cca e.g. [False]
            cca_components e.g. [None]

    Returns
    -------
    all_experimental_groups : dict
        Dictionary containing three nested dictionaries corresponding to the
        parameters for the following experimental conditions

    """

    filename = 'all_experimental_groups.json'

    try:
        exp_groups_nested_dict = json_load(filename, ordered=True)

    except FileNotFoundError as e:
        exp_groups_nested_dict = OrderedDict()
        i = 0
        for key, x_group in all_params.items():
            # c = ascii_uppercase[i]
            # ordered_keys = [c + '_' + x for x in params.keys()]
            all_combinations = [OrderedDict(zip(x_group.keys(), x)) for x in it.product(*x_group.values())]

            for params in all_combinations:
                params['format_str'] = gen_params_str(**params)

            exp_groups_nested_dict[key] = all_combinations

            i += 0

        # Save to reference file
        json_dump(exp_groups_nested_dict,
                  filename,
                  ordered=True, )

    final_experimental_groups = []
    for key, value in exp_groups_nested_dict.items():
        final_experimental_groups.extend(value)

    return final_experimental_groups


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test',
                        help='specify whether to run experiments using a small portion of the dataset',
                        type=int,
                        default=100,
                        )

    args = parser.parse_args()

    os.chdir(config.project_dir)  # Parent directory for project

    # Specify whether test or no test
    if args.test:
        subset_i = (0, 1)
    else:
        subset_i = (0, -1)

    # Loads data from file
    os.chdir(config.full_datasets_path)  # Directory for final, processed corpora
    all_data = load_datasets()

    # Creates directory for results datasets (e.g. feature matrices etc.)
    make_directory(config.results_datasets)
    os.chdir(config.results_datasets)

    # Generates dictionaries containing parameters for each experimental condition
    all_experiments = gen_experimental_groups(**config.all_params)[subset_i[0]:subset_i[1]]
    all_experiments_df = pd.concat([pd.DataFrame(x, index=[i]) for i, x in enumerate(all_experiments)])
    print('\nAll experimental groups: \n{}\n'.format(all_experiments_df))

    all_results = OrderedDict()
    all_results_dfs = []
    all_correct_links = []
    for i, params_dict in enumerate(all_experiments):

        print('\nHyperparameters for next experimental group:\n{}'.format(params_dict))

        # VOCABULARY GENERATION #
        vocab = ft_x.vocab_generation(all_data['full_pm'],
                                      all_data['full_web'],
                                      all_data['full_web_pm_corpus'],
                                      **params_dict, )

        # FEATURE EXTRACTION #
        # Represents web and PubMed documents in the vector space (generates feature matrices
        os.chdir(config.results_datasets)  # Directory for feature matrices, rankings etc.
        vectorizer, feature_matrix = ft_x.feature_representation(vocab,
                                                                 all_data['full_web_pm_corpus'],
                                                                 **params_dict, )

        # DIMENSIONALITY REDUCTION USING TSVD (WHERE REQUIRED) #
        reduced_feature_matrix = ft_x.perform_dimensionality_reduction(feature_matrix,
                                                                       **params_dict, )


        # FEATURE TRANSFORMATION WITH CCA (WHERE REQUIRED) #
        web_test_matrix_final, pm_test_matrix_final = cca.perform_cca(reduced_feature_matrix,
                                                                      all_data['corpus_lengths'],
                                                                      **params_dict, )

        # DOCUMENT SIMILARITY AND RANKING #
        correct_links_ranks = doc_sim_rank.perform_cosine_similiarity(web_test_matrix_final,
                                                                      pm_test_matrix_final,
                                                                      all_data['web_corpus_test'],
                                                                      all_data['pm_corpus_test'],
                                                                      all_data['known_links_test'],
                                                                      **params_dict, )
        all_correct_links.append(correct_links_ranks)

        # EVALUATION METRICS #
        results_dict = doc_sim_rank.measures(correct_links_ranks,
                                             **params_dict, )

        print('\n' + str(results_dict) + '\n')
        all_results[params_dict['format_str']] = results_dict
        all_results_dfs.append(pd.DataFrame(results_dict,
                                            index=[i], ))

    json_dump(all_results, 'ALL_RESULTS.json')

    final_results = pd.concat(all_results_dfs)
    final_results.to_pickle('ALL_RESULTS.pkl')
    final_results.to_csv('ALL_RESULTS.csv')

    pickle_dump(all_correct_links,
                'ALL_CORRECT_RANKS.pkl', )
