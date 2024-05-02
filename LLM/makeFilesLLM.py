import codecs
import json 
import csv

'''
The gold file needs to be generated from each facets gold file in the csfcube repo. 
e.g. test-pid2anns-csfcube-background.json. The gold file format needs to be a 
tsv file where every line corresponds to a query document pair and its relevance judgement. 
<query pid> 0 <candidate pid> <relevance_adju>
'''


def createGoldFile(facet):
    if facet == "all":
        testFolds = ["1", "2"]
        for test in testFolds:
            qrels_file_path = f"./GoldFiles/{facet}/csfcube-{facet}-fold{test}_test.qrels"
            with open(qrels_file_path, 'w', newline='') as tsv_file:
                for subFacet in ["background", "method", "result"]:
                    filePath = f"../gold/test-pid2anns-csfcube-{subFacet}.json"
                    with codecs.open(filePath, 'r', 'utf-8') as fp:
                        # load gold json of sub-facet
                        qpid2pool = json.load(fp)
                        key = f"fold{test}_test"
                        # query documents from facet2folds
                        
                        # loop over each query document in the fold
                        for qpidFacet in testQPids:
                            # if query document facet is the same as the subFacet
                            if subFacet in qpidFacet:
                                qpid = qpidFacet.replace(f"_{subFacet}", "")
                                # loop over each qpid' candidates in the gold file
                                for i in range(len(qpid2pool[qpid]['cands'])):
                                    # if i == 0:
                                    line = f"{qpidFacet}\t0\t{qpid2pool[qpid]['cands'][i]}\t{qpid2pool[qpid]['relevance_adju'][i]}\n"
                                    tsv_file.write(line)
    else:
        filePath = "../gold/test-pid2anns-csfcube-" + facet + ".json"
        # split = "test"
        qrels_file_path = f"./gold-LLM-{facet}_test.qrels"
        # newline to prevent extra newline characters and currently using qrel filename
        with codecs.open(filePath, 'r', 'utf-8') as fp, open(qrels_file_path, 'w', newline='') as tsv_file:
            qpid2pool = json.load(fp)

            for qpid in qpid2pool:
                qpid = qpid.replace("_background", "")
                qpid = qpid.replace("_method", "")
                qpid = qpid.replace("_result", "")
                for i in range(len(qpid2pool[qpid]['cands'])):
                    line = f"{qpid}\t0\t{qpid2pool[qpid]['cands'][i]}\t{qpid2pool[qpid]['relevance_adju'][i]}\n"
                    tsv_file.write(line)
'''
The predictions file needs to be tsv file also. Im sending you a function to write this out. 
The format of the output is: qid, candid, rank positon, similarity, model_name. 
Note that the scores you write out need to be similarities, not distances. 
You can convert a L2 similarity to a distance as explained here: 
https://stats.stackexchange.com/q/53068/55807 – 1/(1 + L2 distance)
'''
def createRankedFile(facet, model, llm):
    if facet == "all":
        for subFacet in ["background", "method", "result"]:
            # filePath = f"../ranked/{model}/test-pid2pool-csfcube-{model}-{subFacet}-ranked.json"
            # filePath = f"../ranked/test-pid2pool-csfcube-{model}-{subFacet}-ranked.json"
            filePath = f"../800k-{subFacet}-{model}.json"
            testFolds = ["1", "2"]
            for test in testFolds:
                # rankedFileName = f"./PredictionFiles/{facet}/test-pid2anns-csfcube-{facet}-fold{test}_test-{model}.tsv"
                rankedFileName = f"./PredictionFiles/{facet}/800k-{facet}-fold{test}_test-{model}.tsv"
                with codecs.open(filePath, 'r', 'utf-8') as fp, open(rankedFileName, 'a', newline='') as tsv_file:
                    # load ranked json of sub-facet
                    qpid2pool = json.load(fp)
                    key = f"fold{test}_test"
                    # query documents from facet2folds
                    testQPids = facet2folds[facet][key]

                    # non 800k
                    # # loop over each query document in the fold
                    # for qpidFacet in testQPids:
                    #     # if query document facet is the same as the subFacet
                    #     if subFacet in qpidFacet:
                    #         qpid = qpidFacet.replace(f"_{subFacet}", "")
                    #         # loop over each qpid' candidates in the ranked file
                    #         for i in range(len(qpid2pool[qpid])):
                    #             similarity = 1 / (1 + qpid2pool[qpid][i][1])
                    #             line = f"{qpidFacet}\tQ0\t{qpid2pool[qpid][i][0]}\t{i+1}\t{similarity}\t{model}\n"
                    #             tsv_file.write(line)
                    # 800k
                    # loop over each query document in the fold
                    for qpidFacet in testQPids:
                        # if query document facet is the same as the subFacet
                        if subFacet in qpidFacet:
                            qpid = qpidFacet.replace(f"_{subFacet}", "")
                            # loop over each qpid' candidates in the ranked file
                            for i in range(1, len(qpid2pool[qpid])):
                                similarity = 1 / (1 + qpid2pool[qpid][i][1])
                                line = f"{qpidFacet}\tQ0\t{qpid2pool[qpid][i][0]}\t{i}\t{similarity}\t{model}\n"
                                tsv_file.write(line)
    else:
        # filePath = f"../ranked/{model}/test-pid2pool-csfcube-{model}-{facet}-ranked.json" Specter
        # filePath = f"../ranked/test-pid2pool-csfcube-{model}-{facet}-ranked.json" Scincl
        # filePath = f"../800kRanked.json"
        filePath = f"./large/qlft5{llm}-{facet}-true-scincl.json"
        # rankedFileName = f"./PredictionFiles/{facet}/test-pid2anns-csfcube-{facet}-fold{test}_test-{model}.tsv"
        rankedFileName = f"./ranked_t5-{llm}_{facet}_true.tsv"
        with codecs.open(filePath, 'r', 'utf-8') as fp, open(rankedFileName, 'w', newline='') as tsv_file:
            qpid2pool = json.load(fp)

            # 800k
            for qpid in qpid2pool:
                qpid = qpid.replace("_background", "")
                qpid = qpid.replace("_method", "")
                qpid = qpid.replace("_result", "")
                if qpid in qpid2pool:
                    for i in range(1, len(qpid2pool[qpid])):
                        similarity = qpid2pool[qpid][i][1]
                        line = f"{qpid}\tQ0\t{qpid2pool[qpid][i][0]}\t{i}\t{similarity}\t{model}\n"
                        tsv_file.write(line)
                else:
                    similarity = 1 / (1 + 0)
                    line = f"{qpid}\tQ0\t{qpid2pool[qpid][i][0]}\t{i}\t{similarity}\t{model}\n"
                    tsv_file.write(line)
    

facets = ["background", "method", "result"] # , "all"
models = ["l"] # "base", "l"
for facet in facets:
    # createGoldFile(facet)
    for model in models:
        createRankedFile(facet, "scincl", model)
