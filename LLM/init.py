# from . import ce_models  # the __init__.py makes sure this is possible to do 
from ce_models import QueryLikelihoodModel
import codecs
import json 

pid2abstract = {}
with codecs.open('../abstracts-csfcube-preds.jsonl', 'r', 'utf-8') as absfile:
    for line in absfile:
        injson = json.loads(line.strip())
        pid2abstract[injson['paper_id']] = injson
print("small json file")

candToAbstract = {}
with open('../abstracts-preds.json', 'r') as file:
    data = json.load(file)
    for key, value in data.items():
        json_data = data[key]
        title = json_data['title']
        abstract = " ".join(json_data['abstract'])
        paper_id = json_data['paper_id']
        candToAbstract[paper_id] = title + " " + abstract
print("big json file")

qpidToRanked = {}

def getCandidates(candArray):
    abstractArr = []
    for pid, score in candArray:
        abstractArr.append(candToAbstract[str(pid)])
    return abstractArr

def run(isFacet, model):
    LLMs = ["qlft5xl"] # "qlft5base", "qlft5l", 
    for LLMmodel in LLMs:
        if not isFacet:
            ql_model = QueryLikelihoodModel(short_model_name=LLMmodel, prompt=f"Sample abstract: INPUTTEXT. Generate a computer science paper abstract similar to the sample abstract.")
        else:
            ql_model = QueryLikelihoodModel(short_model_name=LLMmodel, prompt=f"Sample abstract: INPUTTEXT. Generate the FACET sentences of a computer science paper abstract similar to the sample abstract.")
            facets = ["background", "method", "result"]
            for facet in facets:
                filePath = f"../800k-{facet}-{model}.json"
                with codecs.open(filePath, 'r', 'utf-8') as fp:
                    qpid2pool = json.load(fp)

                    for qpid, candArray in qpid2pool.items():
                        title = pid2abstract[qpid]['title']
                        if not isFacet:
                            abstract = " ".join(pid2abstract[qpid]['abstract'])
                        else:
                            indices = [index for index, value in enumerate(pid2abstract[qpid]["pred_labels"]) if value == f"{facet}_label"]
                            if facet == "background":
                                indicesTwo = [index for index, value in enumerate(pid2abstract[qpid]["pred_labels"]) if value == f"objective_label"]
                                indices += indicesTwo
                            abstract_list = [pid2abstract[qpid]['abstract'][index] for index in indices]
                            abstract = " ".join(abstract_list)
                        inputText = title + abstract
                        cand_texts = getCandidates(candArray)
                        if not isFacet:
                            query_likelihood_scores = ql_model.batched_ql_scores(inputText, cand_texts) # the code batches cand_texts to compute the score
                        else:
                            query_likelihood_scores = ql_model.batched_ql_scores(inputText, cand_texts, facet)
                        # sort cand_texts by query_likelihood_scores to get the re-ranked candidates
                        qidAndScore = []
                        for i in range(len(candArray)):
                            # new edit
                            qidAndScore.append((candArray[i][0], query_likelihood_scores[i]))
                        ranked_pool = list(sorted(qidAndScore, key=lambda cd: cd[1], reverse=True))
                        qpidToRanked[qpid] = ranked_pool[:]
                    
                    qpid2pool_ranked_serializable = {
                        key: [(cpid, float(dist)) for cpid, dist in value]
                        for key, value in qpidToRanked.items()
                    }

                    outputFileName = f"./{LLMmodel}-{facet}-{isFacet}-{model}.json"
                    with open(outputFileName, 'w') as json_file:
                        json.dump(qpid2pool_ranked_serializable, json_file)

# run(False, "scincl")
run(True, "scincl")