import codecs
import json 
import csv


facet2folds = {
    "background": {"fold1_dev": ["3264891_background", "1936997_background", "11844559_background",
                                 "52194540_background", "1791179_background", "6431039_background",
                                 "6173686_background", "7898033_background"],
                   "fold2_dev": ["5764728_background", "10014168_background", "10695055_background",
                                 "929877_background", "1587_background", "51977123_background",
                                 "8781666_background", "189897839_background"],
                   "fold1_test": ["5764728_background", "10014168_background", "10695055_background",
                                  "929877_background", "1587_background", "51977123_background",
                                  "8781666_background", "189897839_background"],
                   "fold2_test": ["3264891_background", "1936997_background", "11844559_background",
                                  "52194540_background", "1791179_background", "6431039_background",
                                  "6173686_background", "7898033_background"]},
    "method": {"fold1_dev": ["189897839_method", "1791179_method", "11310392_method", "2468783_method",
                             "13949438_method", "5270848_method", "52194540_method", "929877_method"],
               "fold2_dev": ["5052952_method", "10010426_method", "102353905_method", "174799296_method",
                             "1198964_method", "53080736_method", "1936997_method", "80628431_method",
                             "53082542_method"],
               "fold1_test": ["5052952_method", "10010426_method", "102353905_method", "174799296_method",
                              "1198964_method", "53080736_method", "1936997_method", "80628431_method",
                              "53082542_method"],
               "fold2_test": ["189897839_method", "1791179_method", "11310392_method", "2468783_method",
                              "13949438_method", "5270848_method", "52194540_method", "929877_method"]},
    "result": {"fold1_dev": ["2090262_result", "174799296_result", "11844559_result", "2468783_result",
                             "1306065_result", "5052952_result", "3264891_result", "8781666_result"],
               "fold2_dev": ["2865563_result", "10052042_result", "11629674_result", "1587_result",
                             "1198964_result", "53080736_result", "2360770_result", "80628431_result",
                             "6431039_result"],
               "fold1_test": ["2865563_result", "10052042_result", "11629674_result", "1587_result",
                              "1198964_result", "53080736_result", "2360770_result", "80628431_result",
                              "6431039_result"],
               "fold2_test": ["2090262_result", "174799296_result", "11844559_result", "2468783_result",
                              "1306065_result", "5052952_result", "3264891_result", "8781666_result"]},
    "all": {"fold1_dev": ["3264891_background", "1936997_background", "11844559_background",
                          "52194540_background", "1791179_background", "6431039_background",
                          "6173686_background", "7898033_background", "189897839_method",
                          "1791179_method", "11310392_method", "2468783_method", "13949438_method",
                          "5270848_method", "52194540_method", "929877_method", "2090262_result",
                          "174799296_result", "11844559_result", "2468783_result", "1306065_result",
                          "5052952_result", "3264891_result", "8781666_result"],
            "fold2_dev": ["5764728_background", "10014168_background", "10695055_background",
                          "929877_background", "1587_background", "51977123_background",
                          "8781666_background", "189897839_background", "5052952_method", "10010426_method",
                          "102353905_method", "174799296_method", "1198964_method", "53080736_method",
                          "1936997_method", "80628431_method", "53082542_method", "2865563_result",
                          "10052042_result", "11629674_result", "1587_result", "1198964_result",
                          "53080736_result", "2360770_result", "80628431_result", "6431039_result"],
            "fold1_test": ["5764728_background", "10014168_background", "10695055_background",
                           "929877_background", "1587_background", "51977123_background", "8781666_background",
                           "189897839_background", "5052952_method", "10010426_method", "102353905_method",
                           "174799296_method", "1198964_method", "53080736_method", "1936997_method",
                           "80628431_method", "53082542_method", "2865563_result", "10052042_result",
                           "11629674_result", "1587_result", "1198964_result", "53080736_result",
                           "2360770_result", "80628431_result", "6431039_result"],
            "fold2_test": ["3264891_background", "1936997_background", "11844559_background",
                           "52194540_background", "1791179_background", "6431039_background",
                           "6173686_background", "7898033_background", "189897839_method", "1791179_method",
                           "11310392_method", "2468783_method", "13949438_method", "5270848_method",
                           "52194540_method", "929877_method", "2090262_result", "174799296_result",
                           "11844559_result", "2468783_result", "1306065_result", "5052952_result",
                           "3264891_result", "8781666_result"]
            }
}


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
                        testQPids = facet2folds[facet][key]
                        
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
    # if facet == "all":
    #     for subFacet in ["background", "method", "result"]:
    #         filePath = f"../gold/test-pid2anns-csfcube-{subFacet}.json"
    #         # testFolds = ["1", "2"]
    #         # for test in testFolds:
    #         test = "1"
    #         qrels_file_path = f"./GoldFiles/{facet}/csfcube-{facet}-fold{test}_test.qrels"
    #         with codecs.open(filePath, 'r', 'utf-8') as fp, open(qrels_file_path, 'a', newline='') as tsv_file:
    #             # load gold json of sub-facet
    #             qpid2pool = json.load(fp)
    #             key = f"fold{test}_test"
    #             # query documents from facet2folds
    #             testQPids = facet2folds[facet][key]
                
    #             # loop over each query document in the fold
    #             for qpid in testQPids:
    #                 # if query document facet is the same as the subFacet
    #                 if subFacet in qpid:
    #                     qpid = qpid.replace(f"_{subFacet}", "")
    #                     # loop over each qpid' candidates in the gold file
    #                     for i in range(len(qpid2pool[qpid]['cands'])):
    #                         # if i == 0:
    #                         line = f"{qpid}\t0\t{qpid2pool[qpid]['cands'][i]}\t{qpid2pool[qpid]['relevance_adju'][i]}\n"
    #                         tsv_file.write(line)
    else:
        filePath = "../gold/test-pid2anns-csfcube-" + facet + ".json"
        # split = "test"
        testFolds = ["1", "2"]
        for test in testFolds:
            qrels_file_path = f"./GoldFiles/{facet}/csfcube-{facet}-fold{test}_test.qrels"
            # newline to prevent extra newline characters and currently using qrel filename
            with codecs.open(filePath, 'r', 'utf-8') as fp, open(qrels_file_path, 'w', newline='') as tsv_file:
                qpid2pool = json.load(fp)
                key = f"fold{test}_test"
                testQPids = facet2folds[facet][key]

                for qpid in testQPids:
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
def createRankedFile(facet, model):
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
        filePath = f"../800k-{facet}-{model}.json"
        testFolds = ["1", "2"]
        for test in testFolds:
            # rankedFileName = f"./PredictionFiles/{facet}/test-pid2anns-csfcube-{facet}-fold{test}_test-{model}.tsv"
            rankedFileName = f"./PredictionFiles/{facet}/800k-{facet}-fold{test}_test-{model}.tsv"
            with codecs.open(filePath, 'r', 'utf-8') as fp, open(rankedFileName, 'w', newline='') as tsv_file:
                qpid2pool = json.load(fp)
                key = f"fold{test}_test"
                testQPids = facet2folds[facet][key]
                
                # non 800k
                # for qpid in testQPids:
                #     qpid = qpid.replace("_background", "")
                #     qpid = qpid.replace("_method", "")
                #     qpid = qpid.replace("_result", "")
                #     if qpid in qpid2pool:
                #         for i in range(len(qpid2pool[qpid])):
                #             similarity = 1 / (1 + qpid2pool[qpid][i][1])
                #             line = f"{qpid}\tQ0\t{qpid2pool[qpid][i][0]}\t{i+1}\t{similarity}\t{model}\n"
                #             tsv_file.write(line)
                #     else:
                #         similarity = 1 / (1 + 0)
                #         line = f"{qpid}\tQ0\t{qpid2pool[qpid][i][0]}\t{i+1}\t{similarity}\t{model}\n"
                #         tsv_file.write(line)

                # 800k
                for qpid in testQPids:
                    qpid = qpid.replace("_background", "")
                    qpid = qpid.replace("_method", "")
                    qpid = qpid.replace("_result", "")
                    if qpid in qpid2pool:
                        for i in range(1, len(qpid2pool[qpid])):
                            similarity = 1 / (1 + qpid2pool[qpid][i][1])
                            line = f"{qpid}\tQ0\t{qpid2pool[qpid][i][0]}\t{i}\t{similarity}\t{model}\n"
                            tsv_file.write(line)
                    else:
                        similarity = 1 / (1 + 0)
                        line = f"{qpid}\tQ0\t{qpid2pool[qpid][i][0]}\t{i}\t{similarity}\t{model}\n"
                        tsv_file.write(line)
    

facets = ["background", "method", "result", "all"] # , "all"
for facet in facets:
    # createGoldFile(facet)
    createRankedFile(facet, "scincl")
    createRankedFile(facet, "specter")
