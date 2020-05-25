"""
Author: Eliza Harrison

This module retrieves the title and abstract text for vaccine-related PubMed
articles in the original dataset file (altmetric_pmids.csv)

"""

import os
import glob
import csv
import pandas as pd
import eutils
import requests
import lxml
import time


# Specifies whether to download a small number of PubMed articles for testing
test = True

# Specifies names for output csv column headers
fieldnames = ['pmid',
              'year',
              'title',
              'abstract',
              'authors',
              'journal',
              'volume',
              'issue',
              'pages',
              'doi',
              'pmc',
              ]


def load_original_pmids():
    """
    Loads PMIDs for vaccine-related PubMed articles from original dataset file
    :return: dataframe containing all original PMIDs
    """

    # Imports original list of PubMed IDs
    original_pmids = pd.read_csv('altmetric_pmids.csv',
                                 header=None,
                                 usecols=[0],
                                 names=['pmid'],
                                 )
    original_pmids.sort_values(by='pmid',
                               inplace=True,
                               )
    original_pmids.reset_index(inplace=True,
                               drop=True,
                               )

    return original_pmids


def get_downloaded_pmids():
    """
    Attempts to load PMIDs for PubMed articles and associated data already downloaded
    via E-Utilities API
    :return: downloaded_pmids: dataframe containing PMIDs
    """

    # Attempts to load previously downloaded PubMed articles from file
    download_files = glob.glob('PubMed_Data_*.csv')
    if len(download_files) > 0:
        print('Importing PubMed article data from download CSVs...')
        downloaded_pmids = pd.concat((pd.read_csv(file,
                                                  skiprows=[0],
                                                  header=None,
                                                  names=fieldnames,
                                                  encoding='utf-8',
                                                  na_values='None',
                                                  )
                                      for file in download_files
                                      ),
                                     ignore_index=True,
                                     ).fillna('n/a')
        print('Import Complete')
        print('Shape:\n{}'.format(downloaded_pmids.shape))
        print('Columns:\n{}'.format(downloaded_pmids.columns))

    else:
        downloaded_pmids = pd.DataFrame([],
                                        columns=['pmid'],
                                        )

    return downloaded_pmids


def eutils_from_df(input_df, chunksize, output_csv):
    """
    Retrieves and saves PubMed article content from PubMed via E-Utilities API to CSV file
    for set of known PMIDs.

    :param input_df: object name for Dataframe containing PMIDs of interest
    :param chunksize: number of PMIDs to pass to API
    :param output_csv: filename for CSV file to which article content will be saved
    :return: CSV file with rows pertaining to article content for each PMID in input_csv.
        columns correspond to fields retrieved via efetch client:
            'PMID', 'Year', 'Title', 'Abstract', 'Authors', 'Journal', 'Volume', 'Issue',
            'Pages', 'DOI', 'PMC'
        list and dataframe containing all PubMed article data successfully retrieved from database
    """

    # Creates generator object containing each row in the input dataframe
    pm_chunks_gen = (input_df[i:i + chunksize] for i in range(0, len(input_df), chunksize))

    # Initialises empty list for compilation of article dictionaries into single container
    pm_article_list = []

    # Initialise eutils client to access NCBI E-Utilities API
    ec = eutils.Client()

    # Open CSV file to which each PubMed IDs downloaded data appended as a new row with specified column names
    with open(output_csv, 'a') as datafile:
        writer = csv.DictWriter(datafile,
                                fieldnames=fieldnames,
                                )
        writer.writeheader()

        # Converts each chunk of PubMed IDs from dataframe to list
        for chunk_count, chunk in zip(range(0, len(input_df)), pm_chunks_gen):
            try:
                index_list = list(chunk.index.values)
                chunk_list = list(chunk['pmid'])
                print('\nChunk No. ' + str(chunk_count))

                # Passes chunk of PubMed IDs to E-Utilities API
                # Returns iterator object containing key data for each PubMed ID
                pm_article_set = iter(ec.efetch(db='pubmed',
                                                id=chunk_list,
                                                )
                                      )

                # Assigns each PubMed ID an index value
                # Iterates over pm_article_set to access data for each individual PubMed ID
                for id_index, id_value in enumerate(chunk_list):
                    print(index_list[id_index], id_value)
                    try:
                        # For each PMID index/value pair, iterates through article set
                        # Aggregates key article attributes for each PubMed ID into dictionary
                        pm_article = next(pm_article_set)
                        pm_article_content = dict(
                            pmid=str(pm_article.pmid),
                            year=str(pm_article.year),
                            title=str(pm_article.title),
                            abstract=str(pm_article.abstract),
                            authors=str(pm_article.authors),
                            journal=str(pm_article.jrnl),
                            volume=str(pm_article.volume),
                            issue=str(pm_article.issue),
                            pages=str(pm_article.pages),
                            doi=str(pm_article.doi),
                            pmc=str(pm_article.pmc),
                        )

                        print(pm_article_content)
                        print(pm_article.pmid + ' - Download from Enterez complete')

                        # Saves dictionary as new item in list for later construction of dataframe
                        pm_article_list.append(pm_article_content)
                        print(pm_article.pmid + ' - Save to list complete')

                        # Writes dictionary to new row of csv file for future reference
                        writer.writerow(pm_article_content)
                        print(pm_article.pmid + ' - Write Data to CSV Complete')

                    # Except statements for content errors
                    except (StopIteration,
                            TypeError,
                            NameError,
                            ValueError,
                            lxml.etree.XMLSyntaxError,
                            eutils.exceptions.EutilsNCBIError,
                            ) as e1:
                        print('Error: ' + str(e1))
                        continue
                    # Except statements for network/connection errors
                    except(TimeoutError,
                           RuntimeError,
                           ConnectionError,
                           ConnectionResetError,
                           eutils.exceptions.EutilsRequestError,
                           requests.exceptions.ConnectionError,
                           ) as e2:
                        print('Error: ' + str(e2))
                        time.sleep(10)
                        continue

            except StopIteration:
                print('All downloads complete')
                break

    # Save list of dictionaries to dataframe & write to CSV file
    pm_article_df = pd.DataFrame.from_records(pm_article_list,
                                              columns=fieldnames,
                                              )
    print('Save to DataFrame complete')
    datafile.close()

    return pm_article_df


def restart():
    # UNSUCCESSFUL RETRIEVAL RE-ATTEMPT
    # Reads all CSV files containing PubMed data into single DataFrame for analysis (all data fields)
    all_pubmed_files = glob.glob('*.csv')

    print('Importing PubMed article data...')
    pubmed_download_1 = pd.concat((pd.read_csv(file,
                                               skiprows=[0],
                                               header=None,
                                               names=fieldnames,
                                               encoding='utf-8',
                                               na_values='None',
                                               )
                                   for file in all_pubmed_files
                                   ),
                                  ignore_index=True,
                                  ).fillna('n/a')
    print('Import Complete')

    print('Shape:\n{}'.format(pubmed_download_1.shape))
    print('Columns:\n{}'.format(pubmed_download_1.columns))

    # Identifies any articles unable to be accessed via PubMed API by populating new boolean column ['Downloaded']
    # Extracts unsuccessful PMIDs to new dataframe for second retrieval attempt
    # Re-attempts to access these articles via E-Utilities API using pre-defined function
    og_pmids['Downloaded'] = og_pmids['PMID'].isin(pubmed_download_1['PMID'])
    print('No. Unsuccessful Downloads: ' + str(len(og_pmids.loc[~og_pmids['Downloaded']])))

    os.chdir('/Users/lizaharrison/PycharmProjects/Predicting_Papers_1/Article_CSVs/Round 2 - Retry')
    retry = og_pmids.loc[og_pmids['Downloaded'] is False, ['pmid']]
    retry.reset_index(inplace=True,
                      drop=True,
                      )

    retry_df = eutils_from_df(retry, 100, 'pm_data_retry_' + str(today) + '.csv')
    print('Number of successful retries: '
          + str(len(retry_df))
          )

    # Appends records successfully retrieved via retry attempt to full dataset
    pubmed_download_2 = pubmed_download_1.append(retry_df)
    pubmed_download_2.reset_index(inplace=True,
                                  drop=True,
                                  )


if __name__ == '__main__':

    # Sets today's date for saving of dynamic file names
    today = pd.Timestamp('today').strftime('%d-%m-%y')

    # DATASET IMPORT #
    # Loads original list of PMIDs
    og_pmids = load_original_pmids()

    # Attempts to load previously downloaded PubMed articles from file
    pm_downloaded_1 = get_downloaded_pmids()
    pm_for_download = og_pmids.loc[~og_pmids['pmid'].isin(pm_downloaded_1['pmid'])]
    print('Number of PubMed articles already downloaded: {}'.format(len(pm_downloaded_1)))
    print('Number of PubMed articles to download: {}'.format(len(pm_for_download)))

    # PUBMED ARTICLE RETRIEVAL (API) #
    # Accesses articles corresponding to each PMID via E-Utilities API
    if test:
        pm_for_download = pm_for_download[0:10]
    pm_data_eutils = eutils_from_df(pm_for_download, 100, 'PubMed_Data_' + str(today) + '.csv')
    print('\nShape:\n{}'.format(pm_data_eutils.shape))
    print('Columns:\n{}'.format(pm_data_eutils.columns))

    # DUPLICATE CHECK #
    # Imports all CSV files containing PubMed data into single DataFrame for analysis (all data fields)
    pm_downloaded_2 = get_downloaded_pmids()

    # Identifies any articles fully duplicated in the PubMed dataset
    # Removes repeats of header row (from CSV file import) and duplicate articles (from multiple download sessions)
    dup_ids = pm_data_eutils.loc[pm_data_eutils.duplicated(subset='pmid',
                                                           keep=False,
                                                           )]
    pm_downloaded_3 = pm_data_eutils.drop_duplicates(subset='pmid',
                                                     )

    print('\nNumber of duplicated PMIDs: {}'.format(len(dup_ids)))
    print('Final number of downloaded PubMed articles: {}'.format(len(pm_downloaded_3)))
    print('Original number of PMIDs: {}'.format(len(og_pmids)))
    print('Articles not retrieved due to error: {}'.format(len(og_pmids) - len(pm_data_eutils)))

    # SAVE TO FILE #
    # Pickles dataframes for later cleansing
    pd.to_pickle(pm_downloaded_3, 'PubMed_Data_Download_All.pkl')
    print('Save to *pkl complete')

    # Saves dataframes to CSV for storage
    pm_downloaded_3.to_csv('_PubMed_Data_Download_All.csv')
    print('Save to *csv complete')
