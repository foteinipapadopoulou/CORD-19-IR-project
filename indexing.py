import pyterrier as pt
from pyterrier_doc2query import Doc2Query, QueryScorer, QueryFilter
from pyterrier_dr import ElectraScorer
from pyterrier_pisa import PisaIndex # needs 3.7> <3.10 python version

"""
The code for indexing using the Doc2Query-- was retrieved and modified 
for our dataset from the pyterrier example notebook.
https://github.com/terrierteam/pyterrier_doc2query/blob/master/examples/doc2query--.ipynb
"""
def load_dataset():
    dataset = pt.get_dataset("irds:cord19/trec-covid")
    return dataset

def index_with_doc2query_hyphen():
    dataset = load_dataset()
    # Initialize a Doc2Query object with a pre-trained Doc2Query model
    # based on t5-base and trained on MS MARCO.
    # It generates the queries
    doc2query = Doc2Query('macavaney/doc2query-t5-base-msmarco', append=False, num_samples=20)
    # The generated queries will be scored with the crystina-z/monoELECTRA_LCE_nneg3 pre-trained model
    # using Electra scorer since it has the best scores in the Doc2Query-- research
    scorer = ElectraScorer('crystina-z/monoELECTRA_LCE_nneg31')

    # indexing
    # Using the PisaIndex (Performant Indexes and Search for Academia)
    # because of efficient query-time perfomance
    # Relevant paper : https://ceur-ws.org/Vol-2409/docker08.pdf
    index = PisaIndex('./doc2query_index.pisa')
    # QueryFilter uses append=True parameter because
    # we want the filtered documents to be appended to the original list of documents.
    # The use threshold t=3.21484375 is to append documents with a score equal or higher than t
    # 3.215  threshold is the relevance score in the top 30% for the ELECTRA scoring model
    # on the MS MACRO dataset. We are using the same threshold as in the Doc2Query-- approach
    # because they received the best perfomance.
    pipeline = doc2query >> QueryScorer(scorer) >> QueryFilter(append=True, t=3.21484375) >> index
    corpus = dataset.get_corpus_iter()
    pipeline.index(corpus)

if __name__ == '__main__':
    # It needs to have JAVA installed and the JAVA_HOME variable set up
    if not pt.started():
        pt.init()
    pt.logging("INFO")
    index_with_doc2query_hyphen()