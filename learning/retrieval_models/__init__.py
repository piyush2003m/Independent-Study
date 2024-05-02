from .learning.retrieval_models import embedding_models

# The class handles batching for you and expects a dict with all the 
# data you want to embed as: dict(doc_id: dict('title': string, 'abstract': list(string)))
# The abstract should contain a list of sentence strings. The class will return pid2docreps, a dict
# which looks like dict(pid: numpy.array) with 1 embedding per paper because specter2_doc is a biencoder
# You can then use the embeddings the same way that you used SciNCL and Specter.
doc_embedder = embedding_models.TextEmbedder(model_name="specter2_doc")
pid2docreps = doc_embedder.encode(all_texts=pid2abstract)

# For aspire the returned embeddings in pid2docreps are multi vector representations so 
# dict(pid: numpy.array) will have as many embeddings as there are sentences in the abstract
# I will discuss how we will use the multi vector embeddings in our call.
# doc_embedder = embedding_models.TextEmbedder(model_name="aspire")
# pid2docreps = doc_embedder.encode(all_texts=pid2abstract)