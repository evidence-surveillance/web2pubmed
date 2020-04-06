"""
Author: Eliza Harrison

Cleanses and prepares webpage and PubMed article data to generate a set of 1:1 links
that forms the final corpus.

"""
import config

import os
from langdetect import detect
import numpy as np
import pandas as pd
import itertools
import re

min_len_threshold = 100


def load_webpages():
    """
    Loads webpage data from original file
    Returns
    -------
    web_data, web_pm_links : tuple
        Contains the raw, unprocessed webpage data and full set of URL to PMID links

    """

    # Imports web article data for web pages with known links to these PMIDs (Source: Altmetric)
    # Encodes data to UTF-8 prior to saving as DataFrame
    with open('select_w_processed_content__w_final_url_.tsv',
              'r',
              encoding='utf-8',
              errors='ignore',
              ) as tsvfile:
        web_data = pd.read_csv(tsvfile,
                               delimiter='\t',
                               usecols=[0, 1, 2],
                               )
    web_data.sort_values(['pmid',
                          'processed_content',
                          'final_url'
                          ]).reset_index(inplace=True)
    print('Number of webpages with known links to vaccine-related PubMed research: {}'.format(len(web_data)))

    web_pm_links = web_data.loc[:, ['pmid', 'final_url']].sort_values('pmid')
    print('Number of URL-PMID pairings in Altmetric data: {}\n'.format(len(web_pm_links)))

    return web_data, web_pm_links


def prep_pubmed(raw_pubmed_data):
    """
     Cleans and pre-processes raw PubMed data. Processing steps include removal of
    articles without title or abstract text or those below minimum length threshold.
    Parameters
    ----------
    raw_pubmed_data : dataframe
        Original PubMed dataset as downloaded via E-Utilities API

    Returns
    -------
    clean_pubmed_data : dataframe
        Processed PubMed dataset

    """

    # Concatenates PubMed article titles and abstracts into single field
    # Forms PubMed article corpus for vectorization
    clean_pubmed_data = raw_pubmed_data.fillna('')
    clean_pubmed_data['corpus_text'] = pd.Series(clean_pubmed_data['title'] + ': ' + clean_pubmed_data['abstract'])

    # Standardises known <null> values for identification of missing title and abstract text
    clean_pubmed_data.replace([r'^n/a$',
                               r'^\s+$',
                               ],
                              np.NaN,
                              regex=True,
                              inplace=True,
                              )

    # Creates column containing length of corpus text
    clean_pubmed_data['text_length'] = clean_pubmed_data['corpus_text'].apply(lambda x: len(x.split()))

    # Removes any records with less than 100 words of text for analysis
    clean_pubmed_data.drop((clean_pubmed_data.index[(clean_pubmed_data['text_length'] < min_len_threshold)]),
                           inplace=True,
                           )
    print('Number PubMed meeting initial inclusion requirements (e.g. length): {}'.format(
        len(clean_pubmed_data)))

    return clean_pubmed_data


def prep_web(raw_webpage_data, clean_pubmed_data):
    """
    Cleans and pre-processes raw webpage data. Processing steps include separation of title
    and content text, removal of webpages with missing title or content text and those
    below minimum length threshold, removal of webpages with duplicate title or content text and
    mapped to the same PMID.
    Parameters
    ----------
    raw_webpage_data : dataframe
        Original, un-processed webpage data
    clean_pubmed_data : dataframe
        PubMed data after initial pre-processing

    Returns
    -------
    clean_webpage_data : dataframe
        Webpage data after initial pre-processing

    """

    # Identifies web records with PMIDs no longer in PubMed dataset following first cleansing steps
    missing_pmids = raw_webpage_data.loc[~raw_webpage_data['pmid'].isin(clean_pubmed_data.index)]
    # Removes web records with missing PMIDs
    clean_webpage_data = raw_webpage_data.drop(missing_pmids.index.values, )
    clean_webpage_data.reset_index(inplace=True,
                                   drop=False,
                                   )
    clean_webpage_data.rename(
        {
            'index': 'web_id',
        },
        axis=1,
        inplace=True, )

    # Removes TITLE and TEXT headings from web text data and replaces with delimiter not found in any records
    processed_content = clean_webpage_data.replace(
        {
            'processed_content':
                {
                    'TITLE:': '',
                    r'\n': '',
                    'TEXT:': r'{-}',
                }
        }, regex=True, )['processed_content']

    # Replaces whitespace character strings with single space
    clean_webpage_data['processed_content'] = clean_webpage_data['processed_content'].apply(
        lambda x: ' '.join(x.split()))

    # Creates new columns for TITLE and TEXT for analysis of content
    clean_webpage_data[['title', 'content']] = processed_content.str.split(r'{-}',
                                                                           n=1,
                                                                           expand=True,
                                                                           )
    print('\nChecking successful delmiiting of webpages: \n{}\n'.format((clean_webpage_data.loc[0:10, ['title',
                                                                                                       'content',
                                                                                                       ]]
                                                                        )))
    # Strips trailing whitespace from all article text fields
    clean_webpage_data['processed_content'] = [str(text).strip(' ') for text in clean_webpage_data['processed_content']]
    clean_webpage_data['title'] = [str(title).strip(' ') for title in clean_webpage_data['title']]
    clean_webpage_data['content'] = [str(content).strip(' ') for content in clean_webpage_data['content']]

    # Identification of web articles with blank title or content (article text) fields
    clean_webpage_data = clean_webpage_data.assign(
        title_nan=(clean_webpage_data['title'].isnull()) |
                  (clean_webpage_data['title'].isin(['', '0', ' ', 'None']))).assign(
        content_nan=(clean_webpage_data['content'].isnull()) |
                    (clean_webpage_data['content'].isin(['', '0', ' ', 'None'])))

    print('Number of webpages w/ missing title + content: {}'.format(
        len(clean_webpage_data.loc[(clean_webpage_data['title_nan']) &
                                   (clean_webpage_data[
                                       'content_nan'])])))

    # Identification of records with less than 100 words of text
    clean_webpage_data['text_lengths'] = clean_webpage_data['processed_content'].apply(lambda x: len(x.split()))
    clean_webpage_data.drop(clean_webpage_data.loc[clean_webpage_data['text_lengths'] < min_len_threshold].index,
                            inplace=True,
                            )
    print('Number of webpages after removal of those below min. length threshold: {}'.format(len(clean_webpage_data)))

    # Removes foreign language webpages
    # https://cloud.google.com/translate/docs/detecting-language#translate_detect_language-python
    # https://github.com/ssut/py-googletrans
    clean_webpage_data['corpus_text'] = pd.Series(clean_webpage_data['title'] + ': ' + clean_webpage_data['content'])

    # Extracts first 200 characters of each web article for language detection
    word_200 = [' '.join(string.split()[0:200]) for string in clean_webpage_data['corpus_text']]
    foreign_lang = []
    for index, string in enumerate(word_200):
        language = detect(string)
        if language != 'en':
            foreign_lang.append(index)
    web_foreign = clean_webpage_data.iloc[foreign_lang]
    clean_webpage_data.drop(web_foreign.index,
                            inplace=True,
                            )
    print('Removal of foreign language webpages complete')

    # Identifies and removes exact duplicate records
    clean_webpage_data['title'] = clean_webpage_data['title'].str.strip(' ')
    clean_webpage_data['content'] = clean_webpage_data['content'].str.strip(' ')
    dup_content_pmid = clean_webpage_data.loc[clean_webpage_data.duplicated(subset=['content',
                                                                                    'pmid',
                                                                                    ],
                                                                            keep=False,
                                                                            )]
    dup_content_only = clean_webpage_data.loc[clean_webpage_data.duplicated(subset=['content'],
                                                                            keep=False,
                                                                            )]
    print('\n{} web records are duplicated in at least content text and PMID'.format((len(dup_content_pmid))))
    print('In total these duplicated records appear {} times in the webpage dataset'.format(len(dup_content_only)))

    clean_webpage_data = clean_webpage_data.drop_duplicates(subset=['content',
                                                                    'pmid',
                                                                    ],
                                                            keep='first',
                                                            )
    clean_webpage_data.drop_duplicates(subset=['title',
                                               'pmid'],
                                       keep='first',
                                       inplace=True,
                                       )

    clean_webpage_data['pmid_count'] = pd.DataFrame(clean_webpage_data.groupby('pmid')['pmid'].transform('size'))
    clean_webpage_data['url_count'] = pd.DataFrame(
        clean_webpage_data.groupby('final_url')['final_url'].transform('size'))

    dup_pmid_subset = clean_webpage_data.loc[clean_webpage_data['pmid_count'] > 1].sort_values(['pmid', 'corpus_text'])
    dup_url_subset = clean_webpage_data.loc[clean_webpage_data['url_count'] > 1].sort_values(
        ['final_url', 'corpus_text'])
    dup_pmid_grouped = dup_pmid_subset.groupby('pmid')
    dup_url_grouped = dup_url_subset.groupby('final_url')

    return clean_webpage_data


def lcs_algorithm(str1, str2):
    """
    Extracts the longest common substring (in words) between two strings.
    SOURCE: Code adapted from
        https://www.bogotobogo.com/python/python_longest_common_substring_lcs_algorithm_generalized_suffix_tree.php

    Parameters
    ----------
    str1 : str
        Input string 1
    str2 : str
        Input string 2

    Returns
    -------
    lcs : str
        The longest common substring between str1 and str2

    """

    # Removes punctuation from string to prevent premature termination of longest common substring
    str1 = re.sub(r'[^\w\s]', '', str1)
    str2 = re.sub(r'[^\w\s]', '', str2)

    # Splits string into tuple of words, to compute lcs by word (vs character)
    str1_words = tuple(word for word in str1.lower().split())
    str2_words = tuple(word for word in str2.lower().split())

    m = len(str1_words)
    n = len(str2_words)

    matrix = [[0] * (n + 1) for i in range(m + 1)]

    longest = 0
    lcs_set = set()

    for i in range(m):

        for j in range(n):

            if str1_words[i] == str2_words[j]:
                x = matrix[i][j] + 1
                matrix[i + 1][j + 1] = x

                if x > longest:
                    longest = x
                    lcs_set = set()
                    lcs_set.add(str1_words[i - x + 1: i + 1])

                else:
                    pass

    lcs = [' '.join(tup) for tup in lcs_set]

    return lcs


def perform_lcs(clean_webpage_data, min_similarity):
    """
    Performs longest common substring analysis between pairs of webpages that share
    the same PMID.

    Parameters
    ----------
    clean_webpage_data : dataframe
        Webpage data after initial pre-processing
    min_similarity : int
        The minimum similarity (%) between two webpages. Corresponds to a percentage
        of the total length of the longest of the two webpages.

    Returns
    -------
    webpage_data_post_lcs : dataframe
        Webpage dataframe in which no two webpages mapped to the same PMID share
        more than 50% shared text.

    """

    lcs_filename = 'lcs_dataframe.pkl'

    try:
        lcs_df = pd.read_pickle(lcs_filename)
        print('\nLongest Common Substring results loaded from file')
    except FileNotFoundError:

        print('\nBeginning Longest Common Substring analysis...')

        # Generates all distinct pairs of series values
        # Initialises empty list in which indices, lcs and %age match can be stored
        lcs_list = [[], [], [], [], []]
        for group_name, group in clean_webpage_data[['corpus_text',
                                                     'pmid',
                                                     ]].groupby('pmid'):
            article_pairs = itertools.combinations(group['corpus_text'].index, 2)
            for pair in article_pairs:
                index_1, index_2 = pair
                if index_1 != index_2:
                    str1 = group.loc[index_1, 'corpus_text']
                    str2 = group.loc[index_2, 'corpus_text']
                    lcs = lcs_algorithm(str1, str2)
                    if len(lcs) > 0:
                        pct = (len(lcs[0]) / max(len(str1), len(str2))) * 100
                        if pct > min_similarity:
                            print('%s - %s Longest common substring > min threshold' % (index_1, index_2))
                            lcs_list[0].append(group_name)
                            lcs_list[1].append(index_1)
                            lcs_list[2].append(index_2)
                            lcs_list[3].append(lcs)
                            lcs_list[4].append(pct)
                        else:
                            pass
                    else:
                        pass
                else:
                    pass
        print('Longest common substring analysis of all records complete\n')

        # Converts to dataframe
        lcs_df = pd.DataFrame(lcs_list).transpose()
        lcs_df.columns = ['pmid',
                          'article_id_1',
                          'article_id_2',
                          'lcs',
                          '%age_common',
                          ]

        # Assigns IDs according to groups of articles mapped to the same PMID
        group_ids = pd.factorize(lcs_df['pmid'])
        lcs_df['group_id'] = group_ids[0]
        lcs_df.set_index(['group_id'], inplace=True)
        lcs_df.reset_index(inplace=True)
        lcs_df.sort_values(['group_id',
                            'article_id_1',
                            'article_id_2', ],
                           inplace=True, )
        lcs_df.to_pickle(lcs_filename)

    lcs_keep = []
    lcs_drop = []
    for pmid, group in lcs_df.groupby('pmid'):
        group = group.drop_duplicates(['article_id_2'])
        ids = group['article_id_1'].append(group['article_id_2']).drop_duplicates()

        if any(int(article_id) in lcs_keep for article_id in ids.values):
            lcs_drop.extend([int(article_id) for article_id in ids if article_id not in lcs_drop])
        else:
            keep = ids.sample(1).values
            # corpus_text = group['corpus_text_2']
            # lengths = [len(text) for text in corpus_text.values]
            # keep = ids.iloc[np.argmax(lengths)]

            if keep not in lcs_keep:
                lcs_keep.extend(keep)
                lcs_drop.extend([int(article_id) for article_id in ids if article_id is not keep])

    keeping = clean_webpage_data.loc[clean_webpage_data.index.isin(lcs_keep),]
    webpage_data_post_lcs = clean_webpage_data.drop(lcs_drop)

    return webpage_data_post_lcs


def select_web_records(for_selection, full_corpus):
    """
    Method for selecting 1:1 links between URLs and PMIDs
    Parameters
    ----------
    for_selection : dataframe
        The subset of the links from which 1:1 links are to be selected. Options include:
            pmid_1_url_1: PMID only appears once in dataset, against a single URL also only
                appearing once in the dataset
            pmid_many_url_1: URL appears only once in dataset, but it is linked to the same PMID as
                at least one other URL
            pmid_1_url_many: URL is mapped to multiple PMIDs, which are each only mapped to a single URL
            pmid_many_url_many: URL appears multiple times against multiple PMIDs, which are in turn
                linked to multiple URLs
    full_corpus : dataframe
        The full links corpus

    Returns
    -------
    full_corpus_updated : dataframe
        Updated full links corpus containing only those

    """
    subset = for_selection.loc[(~for_selection['pmid'].isin(full_corpus['pmid'].values))
                               & (~for_selection['final_url'].isin(full_corpus['final_url'].values))]

    selected_web_ids = []
    selected_urls = []

    for pmid in subset.loc[:, 'pmid'].unique():
        web_ids = subset.loc[subset['pmid'] == pmid].index.values
        urls = subset.loc[subset['pmid'] == pmid, 'final_url'].values
        corpus_text = subset.loc[subset['pmid'] == pmid, 'corpus_text'].values
        text_lengths = [len(text) for text in corpus_text]
        not_in_corpus = True

        while not_in_corpus is True:
            if len(web_ids) > 1:
                i = text_lengths.index(max(text_lengths))  # for selecting longest web article
                # i = random.randint(0, len(web_ids) - 1)  # for selecting random url
            elif len(web_ids) == 1:
                i = 0
            else:
                not_in_corpus = False
                break

            selected_id = web_ids[i]
            selected_url = urls[i]
            selected_corpus_text = corpus_text[i]

            if selected_url in selected_urls or \
                    selected_url in full_corpus['final_url'].values:
                web_ids = [x for x in web_ids if x != selected_id]
                urls = [x for x in urls if x != selected_url]
                corpus_text = [x for x in corpus_text if x != selected_corpus_text]
                text_lengths = [len(text) for text in corpus_text]
                not_in_corpus = True

            else:
                selected_web_ids.append(selected_id)
                selected_urls.append(selected_url)
                not_in_corpus = False
                print('Web record %s added to final corpus (%s)' % (selected_id, selected_url))

    full_corpus_updated = full_corpus.append(subset.loc[selected_web_ids])

    return full_corpus_updated


def select_one_to_one_links(webpage_data_post_lcs):
    """
    Generates final corpus consisting only of 1:1 PMID to URL links.

    pmid_1_url_1: PMID only appears once in dataset, against a single URL also only
        appearing once in the dataset
    pmid_many_url_1: URL appears only once in dataset, but it is linked to the same PMID as
        at least one other URL
    pmid_1_url_many: URL is mapped to multiple PMIDs, which are each only mapped to a single URL
    pmid_many_url_many: URL appears multiple times against multiple PMIDs, which are in turn
        linked to multiple URLs

    1) Get all existing 1:1 records (1 PMID associated with only 1 URL, that URL only associated with that PMID)
    2) For PMIDs with more than one webpage, randomly pick one to include in dataset OR pick longest one
    Each PMID should appear in the dataset only once, against the longest possible webpage

    Parameters
    ----------
    webpage_data_post_lcs : dataframe
        Webpage data as it appears after LCS

    Returns
    -------
    final_corpora_links, final_web_corpus : tuple
        Dataframes containing the final set of 1:1 URL to PMID links, and the corresponding set
        of webpages.

    """

    # Compiles all web articles that appear only once in the dataset
    # Web records with linked PMIDs also appearing only once in dataset (1 : 1 relationship)
    pmid_1_web_1 = webpage_data_post_lcs.loc[
        (webpage_data_post_lcs['pmid_count'] == 1) & (webpage_data_post_lcs['url_count'] == 1)]

    # Web records for which one of the linked PMIDs only appears once in the dataset
    pmid_1_web_many = webpage_data_post_lcs.loc[
        (webpage_data_post_lcs['pmid_count'] == 1) & (webpage_data_post_lcs['url_count'] > 1)]

    # Web records that appear only once in the dataset, but which are linked to a PMID linked to other web records
    pmid_many_web_1 = webpage_data_post_lcs.loc[
        (webpage_data_post_lcs['pmid_count'] > 1) & (webpage_data_post_lcs['url_count'] == 1)]

    # Web records that appear against multiple web records, which also appear against multiple PMIDs
    pmid_many_web_many = webpage_data_post_lcs.loc[
        (webpage_data_post_lcs['pmid_count'] > 1) & (webpage_data_post_lcs['url_count'] > 1)]

    # Stepwise selection of PubMed articles for each web article
    final_web_corpus_1 = pmid_1_web_1
    final_web_corpus_2 = select_web_records(pmid_1_web_many, final_web_corpus_1)
    final_web_corpus_3 = select_web_records(pmid_many_web_1, final_web_corpus_2)
    final_web_corpus_4 = select_web_records(pmid_many_web_many, final_web_corpus_3)

    # Checks for duplicates
    print('Duplicated URLs in final dataset: %s' % final_web_corpus_4.loc[final_web_corpus_4.duplicated('final_url')])
    print('Duplicated PMIDs in final dataset: %s' % final_web_corpus_4.loc[final_web_corpus_4.duplicated('pmid')])
    print('%s records in final web corpus' % len(final_web_corpus_4))

    final_web_corpus = final_web_corpus_4.drop(['processed_content',
                                                'title_nan',
                                                'content_nan',
                                                'text_lengths',
                                                'pmid_count',
                                                'url_count',
                                                ],
                                               axis=1,
                                               )

    # Addition of web ids to PubMed corpus for those 1:1 links to be used for training
    final_corpora_links = final_web_corpus.loc[:, ['web_id',
                                                   'pmid',
                                                   ]].reset_index(drop=True)
    final_corpora_links.columns = ['web_id',
                                   'pmid',
                                   ]
    final_corpora_links.astype(int)
    final_corpora_links.sort_values('web_id',
                                    inplace=True,
                                    )

    return final_corpora_links, final_web_corpus


if __name__ == '__main__':
    # LOAD DATA #
    pm_articles = pd.read_pickle('PubMed_Data_Downloaded_ALL.pkl').set_index('pmid')
    print('\nNumber of PubMed articles downloaded via API: {}'.format(len(pm_articles)))

    webpages_raw, web_pm_links = load_webpages()

    # INITIALISE DIRECTORY FOR FINAL DATASETS #
    final_datasets_dir = config.final_datasets_path
    if not os.path.exists(final_datasets_dir):
        os.makedirs(final_datasets_dir)

    os.chdir(final_datasets_dir)

    # CLEAN & PRE-PROCESS DATA #
    pm_articles_clean = prep_pubmed(pm_articles)
    webpages_clean = prep_web(webpages_raw, pm_articles_clean)

    # LCS ANALYSIS #
    webpages_post_lcs = perform_lcs(webpages_clean, 50)

    # GENERATE FINAL CORPORA #
    # Identifies distinct web articles in the dataset, independent of the PubMed article to which they have been linked
    # Many web articles are mapped to more than one PMID, and as such appear more than once in the full dataset
    # Identifes web records with duplicate (title and article content) independent of URL and PMID
    dup_text_only = webpages_post_lcs.loc[webpages_post_lcs.duplicated(subset=['corpus_text',
                                                                               'title',
                                                                               'content',
                                                                               ],
                                                                       keep='first',
                                                                       )]
    dup_text_url = webpages_post_lcs.loc[webpages_post_lcs.duplicated(subset=['corpus_text',
                                                                              'title',
                                                                              'content',
                                                                              'final_url',
                                                                              ],
                                                                      keep='first',
                                                                      )]
    print('There are {} distinct webpages (URLs) in the web dataset'.format(
        len(webpages_post_lcs.loc[:, 'final_url'].unique())))
    print('There are {} distinct web articles (title and content text) in the web dataset'.format(
        len(webpages_post_lcs) - len(dup_text_only)))
    print('{} distinct PubMed articles are represented in the web dataset'.format(
        len(webpages_post_lcs.loc[:, 'pmid'].unique())))
    print('{} webpages appear more than once in the dataset (independent of PMID and URL)'.format(len(dup_text_only)))
    print('{} webpages appear more than once in the dataset (independent of PMID only)'.format(len(dup_text_url)))

    # final_webpages_distinct: All eligible webpages - distinct based on title/content text and URL with no linked PMIDs
    all_webpages_distinct = webpages_post_lcs[['corpus_text',
                                               'final_url',
                                               'title',
                                               'content',
                                               'processed_content',
                                               ]].drop_duplicates(subset=['corpus_text',
                                                                          'title',
                                                                          'content',
                                                                          ],
                                                                  keep='first',
                                                                  )
    all_webpages_distinct.columns = ['corpus_text',
                                     'final_url',
                                     'title',
                                     'content',
                                     'original',
                                     ]

    # final_web_corpus_w_links: All webpages remaining in the cleansed dataset, including known links to PMIDs
    all_webpages_w_links = webpages_post_lcs[['pmid',
                                              'corpus_text',
                                              'final_url',
                                              'title',
                                              'content',
                                              ]]

    print('\nNumber of distinct articles in web corpus: {}'.format(len(all_webpages_distinct)))
    print('Number of web-PubMed article links: {}\n'.format(len(all_webpages_w_links)))

    # final_links_corpus: 1:1 URL to PMID links corpus
    # final_web_corpus: Data associated with webpages in the final 1:1 links corpus
    final_links_corpus, final_web_corpus = select_one_to_one_links(webpages_post_lcs)
    final_web_corpus.set_index('web_id',
                               drop=True,
                               inplace=True,
                               )
    final_web_corpus.to_pickle('final_web_corpus.pkl')
    final_web_corpus.to_csv('final_web_corpus.csv')
    final_links_corpus.to_pickle('final_web_pubmed_links.pkl')
    final_links_corpus.to_csv('final_web_pubmed_links.csv')

    # final_pm_corpus: Final set of distinct PubMed articles, with no webpage info or links
    final_pm_corpus = pm_articles_clean[['pmid',
                                         'corpus_text',
                                         'title',
                                         'abstract',
                                         'year',
                                         'authors',
                                         'journal',
                                         'volume',
                                         'issue',
                                         'pages',
                                         'doi',
                                         'pmc',
                                         ]]
    final_pm_corpus.set_index('pmid',
                              drop=True,
                              inplace=True,
                              )
    final_pm_corpus.to_pickle('final_pubmed_corpus.pkl')
    final_pm_corpus.to_csv('final_pubmed_corpus.csv')
