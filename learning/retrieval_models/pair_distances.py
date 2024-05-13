"""
Function to compute distances between sets of vectors.
"""
import random
import ot
import numpy as np
from scipy import spatial, special


def rank_candidates(query_embeds, cpid2candidate_embeds, aggregation_method, distr_temp=0.5):
    """
    Given the keyphrase values for the user and sentence embeddings for the candidates return
    :param query_embeds: numpy matrix of size: num_query_sentences x embedding_dimension
    :param cpid2candidate_embeds: dict with string paper id as the key and values which represent sentence embeddings for the paper.
        The sentence embeddings are a numpy matrix of size: num_query_sentences x embedding_dimension
    :param aggregation_method: string; "l2_attention" or "optimal_transport"
    :param distr_temp: float; distribution temperature for optimal transport. Set to 0.5 if query_embeds
        are the facet specific sentence embeddings and 5000 if they are the full abstracts sentences
    """
    # Go over the candidates and compute
    pid2topksim = {}
    for pid in cpid2candidate_embeds:
        sent_reps = cpid2candidate_embeds[pid]
        # Compute pairwise distances for
        if query_embeds.shape[1] != sent_reps.shape[1]:
            print(query_embeds.shape)
            print(sent_reps.shape)
        pair_dists = spatial.distance.cdist(query_embeds, sent_reps)
        # Compute a mask which selects a subset of pair_dists
        if aggregation_method == 'optimal_transport':
            a_distr = special.softmax(-1 * np.min(pair_dists, axis=1))/distr_temp
            b_distr = special.softmax(-1 * np.min(pair_dists, axis=0))/distr_temp
            pair_mask = ot.bregman.sinkhorn_epsilon_scaling(a_distr, b_distr, pair_dists, reg=0.05)
        elif aggregation_method == 'l2_attention':
            pair_mask = special.softmax(-1 * pair_dists)
        wd = np.sum(pair_dists * pair_mask)
        # Convert the distances to a similarity: https://stats.stackexchange.com/q/53068/55807
        pid2topksim[pid] = 1 / (1 + wd)
    
    # Return sorted pids and the corresponding similarities for the pids as a list of tuples
    top_pids = sorted(pid2topksim.items(), key=lambda tu: tu[1], reverse=True)
    return list(top_pids)


if __name__ == '__main__':
    # Test the above function
    query_embeds = np.random.randint(0, 5, (10, 200))
    canid2embeds = {}
    for i in range(5):
        num_sents = random.choice([3, 4, 5, 6])
        canid2embeds[i] = np.random.randint(0, 5, (num_sents, 200))
    ranked = rank_candidates(query_embeds, canid2embeds, aggregation_method='optimal_transport')
    print(ranked)
    ranked = rank_candidates(query_embeds, canid2embeds, aggregation_method='l2_attention')
    print(ranked)
    