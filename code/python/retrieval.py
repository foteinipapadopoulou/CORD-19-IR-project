import argparse

import pyterrier as pt
# It needs to have JAVA installed and the JAVA_HOME variable set up
if not pt.started():
    pt.init()
pt.logging("INFO")

from pyterrier.measures import *

from code.python.indexing import load_dataset, FULL_TREC_COVID_DATASET_NAME


def retrieval(index_ref, loaded_dataset, variant='title'):
    # Preparing the models
    tfidf = pt.BatchRetrieve(index_ref, wmodel="TF_IDF")
    bm25 = pt.BatchRetrieve(index_ref, wmodel="BM25")
    dir_LM = pt.BatchRetrieve(index_ref, wmodel="DirichletLM")

    # Evaluation
    exp = pt.Experiment(
        [tfidf, bm25, dir_LM],
        loaded_dataset.get_topics(variant=variant),
        loaded_dataset.get_qrels(),
        eval_metrics=[P@20,R@20,'map',nDCG@20],
        round = 4,
        names=["TF IDF", "BM25","DirichletLM"])
    return exp


def retrieval_deep_impact(index_ref, loaded_dataset, variant='title'):
    # Preparing the model
    tf = pt.BatchRetrieve(index_ref, wmodel="Tf")

    # Evaluation
    exp = pt.Experiment(
        [tf],
        loaded_dataset.get_topics(variant=variant),
        loaded_dataset.get_qrels(),
        eval_metrics=[P@20,R@20,'map',nDCG@20],
        round = 4,
        names=["Deep Impact"])
    return exp


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some parameters.')

    parser.add_argument('index_path', type=str, help='A path to the index')
    parser.add_argument('--variant', choices=['title', 'description', 'narrative'], default='title',
                        help="Specify the variant for the topics (default = 'title')")

    parser.add_argument('--di', choices=['yes', 'no'], default='no',
                        help="Specify whether you are using deep impact neural model (default = 'no')")

    args = parser.parse_args()
    index_path = args.index_path
    variant = args.variant
    use_deep_impact = args.di

    index = pt.IterDictIndexer(index_path)
    loaded_dataset = load_dataset(FULL_TREC_COVID_DATASET_NAME)
    if use_deep_impact == 'yes':
        print(retrieval_deep_impact(index_ref=index, loaded_dataset=loaded_dataset, variant=variant))
    else:
        print(retrieval(index_ref=index, loaded_dataset=loaded_dataset, variant=variant))

