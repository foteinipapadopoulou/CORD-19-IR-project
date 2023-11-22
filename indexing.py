import pyterrier as pt
# It needs to have JAVA installed and the JAVA_HOME variable set up
if not pt.started():
    pt.init()
pt.logging("INFO")

from pyterrier_doc2query import Doc2Query, QueryScorer, QueryFilter
from pyterrier_dr import ElectraScorer
from pyterrier_pisa import PisaIndex # needs 3.7> <3.10 python version and linux OS

"""
The code for indexing using the Doc2Query-- was retrieved and modified 
for our dataset from the pyterrier example notebook.
https://github.com/terrierteam/pyterrier_doc2query/blob/master/examples/doc2query--.ipynb
"""
TREC_COVID_DATASET_NAME = "irds:cord19/trec-covid"


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

###
def index_with_doc2query_hyphen():
    dataset = load_dataset(TREC_COVID_DATASET_NAME)
    # Initialize a Doc2Query object with a pre-trained Doc2Query model
    # based on t5-base and trained on MS MARCO(default).
    # Generates the queries but we don't append them because we will remove non-relevant queries
    doc2query = Doc2Query(append=False, num_samples=20)
    # The generated queries will be scored with the crystina-z/monoELECTRA_LCE_nneg3 pre-trained model
    # using Electra scorer since it has the best scores in the Doc2Query-- research
    scorer = ElectraScorer('crystina-z/monoELECTRA_LCE_nneg31')

    # Indexing

    # Using the PisaIndex (Performant Indexes and Search for Academia)
    # because of efficient query-time performance
    # Relevant paper : https://ceur-ws.org/Vol-2409/docker08.pdf
    # TODO In order to use it later for retrieval check the documentation: https://github.com/terrierteam/pyterrier_pisa
    # TODO we can also use the pt.DictIndexer
    index = PisaIndex('./doc2query_index.pisa')

    # QueryFilter uses append=True parameter because
    # we want the filtered documents to be appended to the original list of documents.
    # The use threshold t=3.21484375 is to append documents with a score equal or higher than t
    # 3.215  threshold is the relevance score in the top 30% for the ELECTRA scoring model
    # on the MS MACRO dataset. We are using the same threshold as in the Doc2Query-- approach
    # because they received the best performance.
    pipeline = doc2query >> QueryScorer(scorer) >> QueryFilter(append=True, t=3.21484375) >> index
    pipeline.index(dataset.get)


index_with_doc2query_hyphen()