import codecs
import json 
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

pid2abstract = {}

class Model:
    def __init__(self):
        # load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
        self.model = AutoModel.from_pretrained('allenai/specter')

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

# Read in pools for the queries per facet.
with codecs.open('test-pid2anns-csfcube-background.json', 'r', 'utf-8') as fp:
    qpid2pool = json.load(fp)

# Rank the candidates per query.
qpid2pool_ranked = {}
my_model = Model()
for qpid in qpid2pool.keys():
    # Get the paper-ids for candidates.
    cand_pids = qpid2pool[qpid]['cands']
    # Compute the distance between a query and candidate.
    dist_list = my_model.computeDistance(qpid, cand_pids)
    query_cand_distance = [(cpid, dist) for cpid, dist in zip(cand_pids, dist_list)] 
    # Sort the candidates in predicted rank order - smallest to largest distances.
    ranked_pool = list(sorted(query_cand_distance, key=lambda cd: cd[1]))
    qpid2pool_ranked[qpid] = ranked_pool

# with codecs.open('test-pid2pool-csfcube-my_model-background-ranked.json', 'w', 'utf-8') as fp:
#     json.dump(qpid2pool_ranked, fp)

qpid2pool_ranked_serializable = {
    key: [(cpid, float(dist)) for cpid, dist in value]
    for key, value in qpid2pool_ranked.items()
}

# Dump the object into a JSON file
with open('test-pid2pool-csfcube-my_model-background-ranked.json', 'w') as json_file:
    json.dump(qpid2pool_ranked_serializable, json_file)

# with open('test-pid2pool-csfcube-my_model-background-ranked.json', "w") as json_file:
#     # Write the JSON object to the file
#     json.dump(qpid2pool_ranked, json_file)