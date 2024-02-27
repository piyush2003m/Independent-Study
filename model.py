import codecs
import json 
from transformers import AutoTokenizer, AutoModel
import torch
import numpy

pid2abstract = {}

# the following function computes embedding for any pair of query and candidate documents
# and then returns the L2 distance between the two
def my_model(query, candidate):
    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
    model = AutoModel.from_pretrained('allenai/specter')

    query_text = pid2abstract[query]['title'] + tokenizer.sep_token + " ".join(pid2abstract[query]['abstract'])
    cand_text = pid2abstract[candidate]['title'] + tokenizer.sep_token + " ".join(pid2abstract[candidate]['abstract'])

    # preprocess the input
    inputs = tokenizer([query_text, cand_text], padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        result = model(**inputs)

    # take the first token in the batch as the embedding
    query_embedding = result.last_hidden_state[0, 0, :]
    cand_embedding = result.last_hidden_state[1, 0, :]
    return numpy.linalg.norm(query_embedding-cand_embedding)

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
for qpid in qpid2pool.keys():
    # Get the paper-ids for candidates.
    cand_pids = qpid2pool[qpid]['cands']
    # Compute the distance between a query and candidate.
    query_cand_distance = []
    for cpid in cand_pids:
        dist = my_model(qpid, cpid)
        query_cand_distance.append((cpid, dist))
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