import argparse

import pyterrier as pt
# It needs to have JAVA installed and the JAVA_HOME variable set up
if not pt.started():
    pt.init()
pt.logging("INFO")

from pyterrier.measures import *
from pyterrier_doc2query import Doc2Query, QueryScorer, QueryFilter
from pyterrier_dr import ElectraScorer

"""
The code for indexing using the Doc2Query-- was retrieved and modified 
for our dataset from the pyterrier example notebook.
https://github.com/terrierteam/pyterrier_doc2query/blob/master/examples/doc2query--.ipynb
"""

# Setting up some constants for the CORD-19 dataset for each round
FULL_TREC_COVID_DATASET_NAME = "irds:cord19/trec-covid"
ROUND_TREC_COVID_DATASET_NAME = f"{FULL_TREC_COVID_DATASET_NAME}/round"
STAND_INDEX_NAME = 'standard_index_round'
DOC2QUERY_INDEX_NAME = 'doc2query--_index_round'


def load_dataset(dataset_name):
    dataset = pt.get_dataset(dataset_name)
    return dataset

""" 
    Iterating over docs to remove duplicate and empty docs
    Code retrieved by : https://github.com/terrierteam/pyterrier_deepimpact/blob/main/cord19_example.py 
"""
def text_iter(doc_iter):
    encountered_docnos = set()

    for doc in doc_iter:
        # Skipping over empty docs
        if len(doc['title'].strip()) == 0 or len(doc['abstract'].strip()) == 0:
            continue
        # Skipping over duplicate docs and merging fields
        if doc['docno'] not in encountered_docnos:
            yield {"docno": doc['docno'], "text": '{title} {abstract}'.format(**doc)}
            encountered_docnos.add(doc['docno'])

"""
Standard indexing with pyterrier
"""
def indexing(trec_covid_round):
    if trec_covid_round == 0 :
        dataset = FULL_TREC_COVID_DATASET_NAME
    else:
        dataset = f'{ROUND_TREC_COVID_DATASET_NAME}{trec_covid_round}'
    round_dataset = load_dataset(dataset)

    # Creating index cord19
    indexer = pt.IterDictIndexer(f'./indexes/{STAND_INDEX_NAME}{trec_covid_round}')
    index_ref = indexer.index(text_iter(round_dataset.get_corpus_iter()))
    return index_ref, round_dataset

def retrieval(index_ref, round_dataset):
    # Preparing the models
    tfidf = pt.BatchRetrieve(index_ref, wmodel="TF_IDF")
    bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25")
    dir = pt.BatchRetrieve(index_ref, wmodel="DirichletLM")

    # Evaluation
    exp = pt.Experiment(
        [tfidf,bm25,dir],
        round_dataset.get_topics(variant='title'),
        round_dataset.get_qrels(),
        eval_metrics=[P@20,R@20,'map',nDCG@20],
        round = 4,
        names=["TF_IDF", "BM25","DirichletLM"])
    return exp


def doc2query_minus_minus_indexing(trec_covid_round):
    #  Initialize a Doc2Query object with a pre-trained Doc2Query model based on t5-base and trained on MS MARCO(default).
    #  It generates the queries but we don't append them because we will remove non-relevant queries
    doc2query = Doc2Query(append=False, num_samples=20)
    # The generated queries will be scored with the "crystina-z/monoELECTRA_LCE_nneg3" pre-trained model
    # using Electra scorer since it has the best scores in the Doc2Query-- research
    scorer = ElectraScorer('crystina-z/monoELECTRA_LCE_nneg31')
    if trec_covid_round == 0:
        dataset = FULL_TREC_COVID_DATASET_NAME
    else:
        dataset = f'{ROUND_TREC_COVID_DATASET_NAME}{trec_covid_round}'
    round_dataset = load_dataset(dataset)

    index = pt.IterDictIndexer(f'./indexes/{DOC2QUERY_INDEX_NAME}{trec_covid_round}')
    pipeline = doc2query >> QueryScorer(scorer) >> QueryFilter(append=True, t=3.21484375) >> index

    index_ref = pipeline.index(text_iter(round_dataset.get_corpus_iter()))
    return index_ref, round_dataset


def doc2query_minus_minus_retrieval(index_ref, round_dataset):
    exp = retrieval(index_ref, round_dataset)
    print(exp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some parameters.')

    parser.add_argument('round', type=int, choices=range(0, 6), help='An integer value from 0 to 5')

    parser.add_argument('--doc2query', choices=['yes', 'no'], default='no',
                        help='Specify yes or no for doc2query (default is no)')

    args = parser.parse_args()

    round_dataset = args.round
    use_doc2query = args.doc2query

    if use_doc2query == "yes":
        index_ref, round_dataset = doc2query_minus_minus_indexing(round_dataset)
        doc2query_minus_minus_retrieval(index_ref, round_dataset)
    else:
        index_ref, round_dataset = indexing(round_dataset)
        exp = retrieval(index_ref, round_dataset)
        print(exp)