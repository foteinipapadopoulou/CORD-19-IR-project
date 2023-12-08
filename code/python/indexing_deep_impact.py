import pyterrier as pt
# It needs to have JAVA installed and the JAVA_HOME variable set up
if not pt.started():
    pt.init()
pt.logging("INFO")

from pyt_deepimpact import DeepImpactIndexer
from code.python.indexing import load_dataset, FULL_TREC_COVID_DATASET_NAME, text_iter
from code.python.retrieval import retrieval_deep_impact

DEEPIMPACT_INDEX_NAME = 'deepimpact_index'

"""
The code for Deep Impact neural model was retrieved and modified 
for our dataset from the pyterrier example notebook.
https://github.com/terrierteam/pyterrier_deepimpact/blob/main/pyt_deepimpact_vaswani.ipynb
"""
def deep_impact_indexing():
    loaded_dataset = load_dataset(FULL_TREC_COVID_DATASET_NAME)

    index_path = f'./indexes/{DEEPIMPACT_INDEX_NAME}'

    parent = pt.IterDictIndexer(index_path)
    parent.setProperty("termpipelines", "")

    # Set base model with 'gsarti/covidbert-nli' pre-trained model on CORD-19 dataset
    # https://huggingface.co/gsarti/covidbert-nli
    indexer = DeepImpactIndexer(parent, batch_size=32, base_model='gsarti/covidbert-nli')
    indexer.index(text_iter(loaded_dataset.get_corpus_iter()))

    index_ref = pt.IndexRef.of(index_path + "/data.properties")
    index_di = pt.IndexFactory.of(index_ref)
    return index_di, loaded_dataset


if __name__ == '__main__':
    index, loaded_dataset = deep_impact_indexing()
    print(retrieval_deep_impact(index, loaded_dataset))