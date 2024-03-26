import codecs
import json 
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

pid2abstract = {}

class Model:
    def __init__(self, model):
        # load model and tokenizer
        if model == "specter":
            self.tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
            self.model = AutoModel.from_pretrained('allenai/specter')
        elif model == "scincl":
            self.tokenizer = AutoTokenizer.from_pretrained('malteos/scincl')
            self.model = AutoModel.from_pretrained('malteos/scincl')

    def computeDistance(self, query, candidates):
        query_text = pid2abstract[query]['title'] + self.tokenizer.sep_token + " ".join(pid2abstract[query]['abstract'])
        cand_texts = np.array([pid2abstract[cpid]['title'] + self.tokenizer.sep_token + " ".join(pid2abstract[cpid]['abstract']) for cpid in candidates])

        tokenize_input = np.concatenate(([query_text], cand_texts))
        # preprocess the input
        inputs = self.tokenizer(tokenize_input.tolist(), padding=True, truncation=True, return_tensors="pt", max_length=512)
        with torch.no_grad():
            result = self.model(**inputs)

        # first token in the query and 
        query_embedding = result.last_hidden_state[0, 0, :]
        cand_embeddings = result.last_hidden_state[1:, 0, :]
        
        # Compute Euclidean distance between query embedding and each candidate embedding
        distances = np.linalg.norm(query_embedding.unsqueeze(0) - cand_embeddings, axis=1)
        
        return distances.tolist()

# Similar to what I did in the earlier explore.py file. For each paper id, its information
# is stored as value in the dictionary. 
with codecs.open('abstracts-csfcube-preds.jsonl', 'r', 'utf-8') as absfile:
    for line in absfile:
        injson = json.loads(line.strip())
        pid2abstract[injson['paper_id']] = injson

facets = ["background", "method", "result"]
models = ["specter"] # "scincl", 

for model in models:
    for facet in facets:
        fileName = f"gold/test-pid2anns-csfcube-{facet}.json"
        with codecs.open(fileName, 'r', 'utf-8') as fp:
            qpid2pool = json.load(fp)
        # Rank the candidates per query.
        qpid2pool_ranked = {}
        my_model = Model(model)
        for qpid in qpid2pool.keys():
            # Get the paper-ids for candidates
            cand_pids = qpid2pool[qpid]['cands']
            # Compute the distance between a query and candidate.
            dist_list = my_model.computeDistance(qpid, cand_pids)
            query_cand_distance = [(cpid, dist) for cpid, dist in zip(cand_pids, dist_list)] 
            # Sort the candidates in predicted rank order - smallest to largest distances.
            ranked_pool = list(sorted(query_cand_distance, key=lambda cd: cd[1]))
            qpid2pool_ranked[qpid] = ranked_pool

        qpid2pool_ranked_serializable = {
            key: [(cpid, float(dist)) for cpid, dist in value]
            for key, value in qpid2pool_ranked.items()
        }

        outputFileName = f"ranked/{model}/test-pid2pool-csfcube-{model}-{facet}-ranked.json"
        with open(outputFileName, 'w') as json_file:
            json.dump(qpid2pool_ranked_serializable, json_file)

# Background
# with codecs.open('gold/test-pid2anns-csfcube-background.json', 'r', 'utf-8') as fp:
#     qpid2pool = json.load(fp)

# Method
# with codecs.open('gold/test-pid2anns-csfcube-method.json', 'r', 'utf-8') as fp:
#     qpid2pool = json.load(fp)

# Result
# with codecs.open('gold/test-pid2anns-csfcube-result.json', 'r', 'utf-8') as fp:
#     qpid2pool = json.load(fp)

# Rank the candidates per query.
# qpid2pool_ranked = {}
# my_model = Model("scincl")
# for qpid in qpid2pool.keys():
#     # Get the paper-ids for candidates.
#     cand_pids = qpid2pool[qpid]['cands']
#     # Compute the distance between a query and candidate.
#     dist_list = my_model.computeDistance(qpid, cand_pids)
#     query_cand_distance = [(cpid, dist) for cpid, dist in zip(cand_pids, dist_list)] 
#     # Sort the candidates in predicted rank order - smallest to largest distances.
#     ranked_pool = list(sorted(query_cand_distance, key=lambda cd: cd[1]))
#     qpid2pool_ranked[qpid] = ranked_pool

# qpid2pool_ranked_serializable = {
#     key: [(cpid, float(dist)) for cpid, dist in value]
#     for key, value in qpid2pool_ranked.items()
# }

# Dump the object into a JSON file
# Specter
# Background
# with open('ranked/test-pid2pool-csfcube-specter-background-ranked.json', 'w') as json_file:
#     json.dump(qpid2pool_ranked_serializable, json_file)

# Method
# with open('ranked/test-pid2pool-csfcube-specter-method-ranked.json', 'w') as json_file:
#     json.dump(qpid2pool_ranked_serializable, json_file)

# Result
# with open('ranked/test-pid2pool-csfcube-specter-result-ranked.json', 'w') as json_file:
#     json.dump(qpid2pool_ranked_serializable, json_file)

# Scincl
# Background
# with open('ranked/test-pid2pool-csfcube-scincl-background-ranked.json', 'w') as json_file:
#     json.dump(qpid2pool_ranked_serializable, json_file)

# Method
# with open('ranked/test-pid2pool-csfcube-scincl-method-ranked.json', 'w') as json_file:
#     json.dump(qpid2pool_ranked_serializable, json_file)