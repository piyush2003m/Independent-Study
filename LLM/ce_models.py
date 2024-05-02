"""
Clases implementing query likelihood or supervised crossencoders.
"""
import collections
import re

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

name2rrmodel = {
        'qlft5base': "google/flan-t5-base",
        'qlft5l': "google/flan-t5-large",
        'qlft5xl': "google/flan-t5-xl",
        'qlft5xxl': "google/flan-t5-xxl"
    }


class QueryLikelihoodModel:
    """
    Initialize a reranking model and score query-document pairs with it.
    """
    def __init__(self, short_model_name, prompt):
        """
        :param short_model_name: string; key from the name2rrmodel dict above
        :param prompt: string; the prompt for the input to the encoder of the T5 model.
            The prompt should contain the string INPUTTEXT that the ql_score_with_t5 method will with
            the candidate document text.
            For example; for a classification task:
            - prompt: "Wikipedia Article: INPUTTEXT. Generate a category to describe this Wikipedia Article."
            For the CSFCube unfaceted similarity task;
            - prompt: "Sample abstract: INPUTTEXT. Generate a computer science paper abstract similar to
                    the sample abstract."
            - Note: query_text should be a full query abstract
            For the CSFCube faceted similarity task:
            - prompt: "Sample abstract: INPUTTEXT. Generate the FACET sentences of a computer science paper
                    abstract similar to the sample abstract."
            - Note: query_text is only the query facet text from the query abstract
        """
        self.tokenizer = T5Tokenizer.from_pretrained(name2rrmodel[short_model_name])
        self.model = T5ForConditionalGeneration.from_pretrained(name2rrmodel[short_model_name], device_map="auto",
                                                                cache_dir='/gypsum/work1/zamani/psmaheshwari/tmp')
        # ql_score_with_t5 scoring will replace INPUTTEXT with the input text
        self.prompt = prompt
    
    def ql_score_with_t5(self, query_text, input_texts, facet):
        """
        Given a T5 model score the input texts for relevance to the query text with the query likelihood.
        :param query_text: string; query text is repeated in this function; batching happens for input_texts.
        :param input_texts: list(string); the candidate documents to compute a score for
        """
        if facet is None:
            doc_texts = [self.prompt.replace("INPUTTEXT", d) for d in input_texts]
        else:
            doc_texts = [self.prompt.replace("INPUTTEXT", d).replace("FACET", facet) for d in input_texts]
        # doc_texts = [re.sub('INPUTTEXT', d, self.prompt) for d in input_texts]
        # doc_texts = [re.sub('INPUTTEXT', d, self.prompt.replace('\\e', 'e')) for d in input_texts]
        input_encodings = self.tokenizer(doc_texts, padding='longest', truncation=True, return_tensors='pt')
        input_tok_ids, input_att_mask = input_encodings.input_ids.to('cuda'), input_encodings.attention_mask.to('cuda')
        target_encoding = self.tokenizer(query_text, truncation=True, return_tensors='pt')
        target_tok_ids = target_encoding.input_ids.to('cuda')
        input_len = target_encoding.input_ids.shape[1]
        repeated_target_ids = torch.repeat_interleave(target_tok_ids, len(doc_texts), dim=0)
        # Pass through the model
        with torch.no_grad():
            outs = self.model(input_ids=input_tok_ids, attention_mask=input_att_mask,
                              labels=repeated_target_ids)
        logits = outs.logits
        log_softmax = torch.nn.functional.log_softmax(logits, dim=2)
        loglikelihoods = log_softmax.gather(2, repeated_target_ids.unsqueeze(2)).squeeze(2)
        seq_ll = torch.sum(loglikelihoods, dim=1) / input_len
        doc_text_lls = seq_ll.cpu().tolist()
        # print(torch.cuda.mem_get_info())
        return doc_text_lls
    
    def batched_ql_scores(self, query_text, cand_texts, facet=None, batch_size=32):
        """
        Return a list of likelihood scores for the candidate wrt the query text. - so higher is better
        :param query_text: string;
        :param cand_texts: list(string); the candidate documents to compute a score for
        :param batch_size: int; batch size for batching; reduce if CUDA memory gets exceeded with 48.
        """
        all_log_likelihoods = []
        cand_batch = []
        for cand in cand_texts:
            cand_batch.append(cand)
            if len(cand_batch) == batch_size:
                if len(all_log_likelihoods) % (batch_size*10) == 0:
                    print(f'Scoring: {len(all_log_likelihoods)}/{len(cand_texts)}')
                log_likelihoods = self.ql_score_with_t5(query_text, cand_batch, facet)
                all_log_likelihoods.extend(log_likelihoods)
                cand_batch = []
        
        # Handle the final batch.
        if cand_batch:
            log_likelihoods = self.ql_score_with_t5(query_text, cand_batch, facet)
            all_log_likelihoods.extend(log_likelihoods)
        assert (len(cand_texts) == len(all_log_likelihoods))
        return all_log_likelihoods
    
    def all_query_reranking_scores(self, qid2texts):
        """
        Given the queryid, query text and cand texts compute scores for all the queries.
        - This method is unused.
        :param qid2texts: dict(query_id: {'query_text': string, 'cand_texts': list(string)}
        """
        qid2scores = collections.OrderedDict()
        for qid in qid2texts:
            print(f'Reranking: {qid}')
            scores = self.batched_ql_scores(query_text=qid2texts[qid]['query_text'],
                                            cand_texts=qid2texts[qid]['cand_texts'])
            qid2scores[qid] = scores
        return qid2scores
