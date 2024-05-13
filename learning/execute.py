from retrieval_models import embedding_models
import json
from sklearn.neighbors import NearestNeighbors
import codecs
import collections
import numpy as np

# The class handles batching for you and expects a dict with all the 
# data you want to embed as: dict(doc_id: dict('title': string, 'abstract': list(string)))
# The abstract should contain a list of sentence strings. The class will return pid2docreps, a dict
# which looks like dict(pid: numpy.array) with 1 embedding per paper because specter2_doc is a biencoder
# You can then use the embeddings the same way that you used SciNCL and Specter.

smallerPid2Abstract = {}
with codecs.open('../abstracts-csfcube-preds.jsonl', 'r', 'utf-8') as absfile:
    for line in absfile:
        injson = json.loads(line.strip())
        smallerPid2Abstract[injson['paper_id']] = injson

pid2abstract = {}
pid2abstract = collections.OrderedDict()
int_idx2pid = {}

with open('../abstracts-preds.json', 'r') as file:
    data = json.load(file)
    # embed all the documents
    i = 0
    for key, value in data.items():
        json_data = data[key]
        paper_id = json_data['paper_id']
        title = json_data['title']
        abstract = json_data['abstract']
        pid2abstract[paper_id] = {'title': title, 'abstract': abstract}
        int_idx2pid[i] = paper_id
        i += 1
        # break at 10k documents
        # if i == 500:
        #    break
    
    # testing
    # fileName = f"../gold/test-pid2anns-csfcube-background.json"
    # with codecs.open(fileName, 'r', 'utf-8') as fp:
    #     qpid2pool = json.load(fp)
    #     for qpid in qpid2pool.keys():
    #         pid2abstract[qpid] = {'title': smallerPid2Abstract[qpid]['title'], 'abstract': smallerPid2Abstract[qpid]['abstract']}
    #         int_idx2pid[i] = qpid
    #         i += 1

doc_embedder = embedding_models.TextEmbedder(model_name="specter2_doc")
pid2docreps = doc_embedder.encode(all_texts=pid2abstract)

pid2docreps_copy = {}
for key, value in pid2docreps.items():
    pid2docreps_copy[key] = value.tolist()

# Save doc_embeds to a JSON file
with open('doc_embeds_specter2.json', 'w') as json_file:
    json.dump(pid2docreps_copy, json_file)

# document_neighbor_index = NearestNeighbors(n_neighbors=200, metric='minkowski', p=2)
# document_neighbor_index.fit(embeddings)

document_neighbor_index = NearestNeighbors(n_neighbors=200, metric='minkowski', p=2)

embeddings = np.array(list(pid2docreps.values()))
embeddings = np.squeeze(embeddings)
embeddings_list = embeddings.tolist()
document_neighbor_index.fit(embeddings_list)

facets = ["background", "method", "result"] # 

resJSON = {}
for facet in facets:
    fileName = f"../gold/test-pid2anns-csfcube-{facet}.json"
    with codecs.open(fileName, 'r', 'utf-8') as fp:
        qpid2pool = json.load(fp)
        facetDict = {}
        facetDict = collections.OrderedDict()
        query_ids = []
        for qpid in qpid2pool.keys():
            facetDict[qpid] = {'title': smallerPid2Abstract[qpid]['title'], 'abstract': smallerPid2Abstract[qpid]['abstract']}
            query_ids.append(qpid)
        returnedDict = doc_embedder.encode(all_texts=facetDict)

        queryEmbeds = np.array(list(returnedDict.values()))
        queryEmbeds = np.squeeze(queryEmbeds)
        smaller_embeddings_list = queryEmbeds.tolist()
        neighbor_distances, neighbor_indexes = document_neighbor_index.kneighbors(smaller_embeddings_list)        

        for i in range(len(query_ids)):
            neighbor_indexes[i] = [int_idx2pid[elem] for elem in neighbor_indexes[i]]
            IndexAndDist = list(zip(neighbor_indexes[i], neighbor_distances[i]))
            ranked_pool = list(sorted(IndexAndDist, key=lambda x: x[1]))
            resJSON[query_ids[i]] = ranked_pool
        
        resJSON_serializable = {
            key: [(int(cpid), float(dist)) for cpid, dist in value]
            for key, value in resJSON.items()
        }

        outputFileName = f"800k-{facet}-specter2.json"
        with open(outputFileName, 'w') as json_file:
            json.dump(resJSON_serializable, json_file)




# For aspire the returned embeddings in pid2docreps are multi vector representations so 
# dict(pid: numpy.array) will have as many embeddings as there are sentences in the abstract
# I will discuss how we will use the multi vector embeddings in our call.
# doc_embedder = embedding_models.TextEmbedder(model_name="aspire")
# pid2docreps = doc_embedder.encode(all_texts=pid2abstract)