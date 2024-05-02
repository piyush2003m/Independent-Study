import codecs
import json 
import numpy as np

def getRel(facet, fold, model):
    rankedFileName = f"./PredictionFiles/{facet}/800k-{facet}-fold{fold}_test-{model}.tsv"
    goldFilePath = f"./GoldFiles/{facet}/csfcube-{facet}-fold{fold}_test.qrels"
    visitedQueries = set()
    rel20 = []
    rel200 = []
    with codecs.open(rankedFileName, 'r') as rf, open(goldFilePath, 'r') as gf:
        for line in rf:
            tokens = line.split()
            qid = tokens[0]
            docid = tokens[2]
            if qid not in visitedQueries:
                visitedQueries.add(qid)
                rel = 0
                i = 0
            if checkPresence(qid, docid, goldFilePath):
                rel += 1
            i += 1
            if i == 20:
                relFraction20 = rel/i
                rel20.append(relFraction20)
            if i == 199:
                relFraction200 = rel/i
                rel200.append(relFraction200)
    return rel20, rel200

def checkPresence(qid, docid, goldFilePath):
    with codecs.open(goldFilePath, 'r', 'utf-8') as gf:
        for line in gf:
            tokens = line.split()
            goldQid = tokens[0]
            goldDocid = tokens[2]
            if int(qid) == int(goldQid) and int(docid) == int(goldDocid):
                return True
        return False

facets = ["background", "method", "result"]
folds = ["1", "2"]
models = ["specter", "scincl"]

for model in models:
    for facet in facets:
        for fold in folds:
            rel20, rel200 = getRel(facet, fold, model)
            min_rel20 = np.min(rel20)
            max_rel20 = np.max(rel20)
            median_rel20 = np.median(rel20)

            min_rel200 = np.min(rel200)
            max_rel200 = np.max(rel200)
            median_rel200 = np.median(rel200)

            # Print the results
            print(f"{model} {facet} {fold} 20: Min:{min_rel20} Max:{max_rel20} Median:{median_rel20}")
            print(f"{model} {facet} {fold} 200: Min:{min_rel200} Max:{max_rel200} Median:{median_rel200}")
            print(f"{model} {facet} {fold} 20:{np.mean(rel20)} 200:{np.mean(rel200)}")

