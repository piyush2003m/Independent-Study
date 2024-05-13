from retrieval_models import embedding_models
import json
from sklearn.neighbors import NearestNeighbors
import codecs
import collections
import numpy as np
from retrieval_models import pair_distances

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


facets = ["background", "method", "result"]
for facet in facets:
    inputDict = collections.OrderedDict()
    filePath = f"../800k-{facet}-scincl.json"
    with codecs.open(filePath, 'r', 'utf-8') as fp:
        qpid2pool = json.load(fp)

        for qpid, candArray in qpid2pool.items():
            for cand, dist in candArray:
                if cand not in inputDict:
                    inputDict[cand] = pid2abstract[str(cand)]
                if qpid not in inputDict:
                    inputDict[qpid] = {'title': smallerPid2Abstract[qpid]['title'], 'abstract': smallerPid2Abstract[qpid]['abstract']}

    # For aspire the returned embeddings in pid2docreps are multi vector representations so 
    # dict(pid: numpy.array) will have as many embeddings as there are sentences in the abstract
    # I will discuss how we will use the multi vector embeddings in our call.
    doc_embedder = embedding_models.TextEmbedder(model_name="aspire")
    pid2docreps = doc_embedder.encode(all_texts=inputDict)

    returnDict = {}
    with codecs.open(filePath, 'r', 'utf-8') as fp:
        qpid2pool = json.load(fp)
        for qpid, candArray in qpid2pool.items():
            # query setup
            queryEmbed = pid2docreps[qpid]
            indices = [index for index, value in enumerate(smallerPid2Abstract[qpid]["pred_labels"]) if value == f"{facet}_label"]
            if facet == "background":
                indicesTwo = [index for index, value in enumerate(smallerPid2Abstract[qpid]["pred_labels"]) if value == f"objective_label"]
                indices += indicesTwo
            
            if len(indices) == 0:
                continue

            revised_query_embeds = np.array([queryEmbed[index] for index in indices])

            # candidates setup
            cand_dict = {}
            for cand, dist in candArray:
                cand_dict[cand] = pid2docreps[cand]
                # if pid2docreps[cand].ndim == 1:
                #     print("cand_dict", pid2docreps[cand].shape)
                #     print("revised query embds", revised_query_embeds.shape)
                #     cand_dict[cand] = pid2docreps[cand][np.newaxis, :]
                # else:
                #     cand_dict[cand] = pid2docreps[cand]
            
            re_ranked = pair_distances.rank_candidates(revised_query_embeds, cand_dict, "l2_attention")
            returnDict[qpid] = re_ranked
    
    resJSON_serializable = {
        key: [(int(cpid), float(dist)) for cpid, dist in value]
        for key, value in returnDict.items()
    }

    outputFileName = f"aspire-{facet}.json"
    with open(outputFileName, 'w') as json_file:
        json.dump(resJSON_serializable, json_file)