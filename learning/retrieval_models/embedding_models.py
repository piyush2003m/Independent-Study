"""
Classes for biencoder and multi vector models which are not supported by
simple biencoders available from SentenceTransformers.
"""
import collections
import os
import json
import codecs

import numpy as np
import torch
from torch import nn as nn
from torch.autograd import Variable
from transformers import AutoModel, AutoTokenizer
from adapters import AutoAdapterModel

from .. import batchers


class ContextualSentenceEmbedder(nn.Module):
    """
    Pass abstracts through Transformer LM, get contextualized sentence reps by averaging contextual word embeddings.
    This is copied over and modified from: src.learning.facetid_models.disent_models.WordSentAlignBiEnc
    """
    def __init__(self, model_hparams):
        """
        :param model_hparams: dict; model hyperparams.
        """
        torch.nn.Module.__init__(self)
        self.bert_encoding_dim = 768  # bert_config.hidden_size or DistilBertConfig.dim
        self.bert_layer_count = 12 + 1  # plus 1 for the bottom most layer.
        self.bert_encoder = AutoModel.from_pretrained(model_hparams['base-pt-layer'])
        self.bert_encoder.config.output_hidden_states = True
    
    def encode(self, batch_dict):
        """
        Function used at test time.
        batch_dict: dict of the form:
            {'bert_batch': dict(); The batch which BERT inputs with flattened and
                concated sentences from query abstracts; Tokenized and int mapped
                sentences and other inputs to BERT.
            'abs_lens': list(int); Number of sentences in query abs.
            'senttok_idxs': list(list(list(int))); batch_size(num_abs_sents(
                    num_sent_tokens(ints)))}
        :return: ret_dict
        """
        doc_bert_batch, doc_abs_lens = batch_dict['bert_batch'], batch_dict['abs_lens']
        doc_query_senttoki = batch_dict['senttok_idxs']
        # Get the representations from the model; batch_size x encoding_dim x max_sents
        _, sent_reps = self.partial_forward(bert_batch=doc_bert_batch, abs_lens=doc_abs_lens,
                                            sent_tok_idxs=doc_query_senttoki)
        # Make numpy arrays and return.
        if torch.cuda.is_available():
            sent_reps = sent_reps.cpu().data.numpy()
        else:
            sent_reps = sent_reps.data.numpy()
        unpadded_sent_reps = []
        for i, num_sents in enumerate(doc_abs_lens):
            # encoding_dim x num_sents
            upsr = sent_reps[i, :, :num_sents]
            # return: # num_sents x encoding_dim
            unpadded_sent_reps.append(upsr.transpose(1, 0))
        # list of batch_size elements each of which is num_sents x encoding_dim
        return unpadded_sent_reps
    
    def partial_forward(self, bert_batch, abs_lens, sent_tok_idxs):
        """
        Pass a batch of sentences through BERT and read off sentence
        representations based on SEP idxs.
        :return:
            doc_cls_reps: batch_size x encoding_dim
            sent_reps: batch_size x encoding_dim x num_sents
        """
        # batch_size x num_sents x encoding_dim
        doc_cls_reps, sent_reps = self.sent_reps_bert(bert_batch=bert_batch, num_sents=abs_lens,
                                                      batch_senttok_idxs=sent_tok_idxs)
        if len(sent_reps.size()) == 2:
            sent_reps = sent_reps.unsqueeze(0)
        if len(doc_cls_reps.size()) == 1:
            doc_cls_reps = doc_cls_reps.unsqueeze(0)
        # Similarity function expects: batch_size x encoding_dim x q_max_sents;
        return doc_cls_reps, sent_reps.permute(0, 2, 1)
    
    def sent_reps_bert(self, bert_batch, batch_senttok_idxs, num_sents):
        """
        Pass the concated abstract through BERT, and average token reps to get sentence reps.
        -- NO weighted combine across layers.
        :param bert_batch: dict('tokid_tt', 'seg_tt', 'attnmask_tt', 'seq_lens'); items to use for getting BERT
            representations. The sentence mapped to BERT vocab and appropriately padded.
        :param batch_senttok_idxs: list(list(list(int))); batch_size([num_sents_per_abs[num_tokens_in_sent]])
        :param num_sents: list(int); number of sentences in each example in the batch passed.
        :return:
            doc_cls_reps: FloatTensor [batch_size x bert_encoding_dim]
            sent_reps: FloatTensor [batch_size x num_sents x bert_encoding_dim]
        """
        seq_lens = bert_batch['seq_lens']
        batch_size, max_seq_len = len(seq_lens), max(seq_lens)
        max_sents = max(num_sents)
        tokid_tt, seg_tt, attnmask_tt = bert_batch['tokid_tt'], bert_batch['seg_tt'], bert_batch['attnmask_tt']
        if torch.cuda.is_available():
            tokid_tt, seg_tt, attnmask_tt = tokid_tt.cuda(), seg_tt.cuda(), attnmask_tt.cuda()
        # Pass input through BERT and return all layer hidden outputs.
        model_outputs = self.bert_encoder(tokid_tt, token_type_ids=seg_tt, attention_mask=attnmask_tt)
        final_hidden_state = model_outputs.last_hidden_state
        # Read of CLS token as document representation.
        doc_cls_reps = final_hidden_state[:, 0, :]
        doc_cls_reps = doc_cls_reps.squeeze()
        # Average token reps for every sentence to get sentence representations.
        # Build the first sent for all batch examples, second sent ... and so on in each iteration below.
        sent_reps = []
        for sent_i in range(max_sents):
            cur_sent_mask = np.zeros((batch_size, max_seq_len, self.bert_encoding_dim))
            # Build a mask for the ith sentence for all the abstracts of the batch.
            for batch_abs_i in range(batch_size):
                abs_sent_idxs = batch_senttok_idxs[batch_abs_i]
                try:
                    sent_i_tok_idxs = abs_sent_idxs[sent_i]
                except IndexError:  # This happens in the case where the abstract has fewer than max sents.
                    sent_i_tok_idxs = []
                cur_sent_mask[batch_abs_i, sent_i_tok_idxs, :] = 1.0
            sent_mask = Variable(torch.FloatTensor(cur_sent_mask))
            if torch.cuda.is_available():
                sent_mask = sent_mask.cuda()
            # batch_size x seq_len x encoding_dim
            sent_tokens = final_hidden_state * sent_mask
            # The sent_masks non zero elements in one slice along embedding dim is the sentence length.
            cur_sent_reps = torch.sum(sent_tokens, dim=1) / \
                            torch.count_nonzero(sent_mask[:, :, 0], dim=1).clamp(min=1).unsqueeze(dim=1)
            sent_reps.append(cur_sent_reps.unsqueeze(dim=1))
        # batch_size x max_sents x encoding_dim
        sent_reps = torch.cat(sent_reps, dim=1)
        return doc_cls_reps, sent_reps


class TextEmbedder:
    """
    Model class to embed texts with models which SentenceTransformer wouldnt support.
    These are:
    - Aspire models: allenai/aspire-contextualsentence-multim-compsci
    - Specter2 models
    """
    def __init__(self, model_name, tok_pooling='cls'):
        self.encoding_dim = 768
        self.model_name = model_name
        self.tok_pooling = tok_pooling
        name2hf_base_name = {
            'aspire': 'allenai/aspire-contextualsentence-multim-compsci',
            # Even though it wasnt traiend for contextual sentence embeddings it seems to work fine
            'mpnet1b_sc': 'sentence-transformers/all-mpnet-base-v2',
            'specter2_doc': 'allenai/specter2_base',
            'specter2_query': 'allenai/specter2_base'
        }
        try:
            hf_name = name2hf_base_name[model_name]
        except KeyError:
            hf_name = model_name
        if model_name in {'aspire', 'mpnet1b_sc'}:
            self.tokenizer = AutoTokenizer.from_pretrained(hf_name)
            self.model = ContextualSentenceEmbedder(model_hparams={'base-pt-layer': hf_name})
        elif model_name == 'specter2_doc':
            self.tokenizer = AutoTokenizer.from_pretrained(hf_name)
            self.model = AutoAdapterModel.from_pretrained(hf_name)
            self.model.load_adapter('allenai/specter2', source="hf", set_active=True)
        elif model_name == 'specter2_query':
            self.tokenizer = AutoTokenizer.from_pretrained(hf_name)
            self.model = AutoAdapterModel.from_pretrained(hf_name)
            self.model.load_adapter('allenai/specter2_adhoc_query', source="hf", set_active=True)
        else:  # Any basic transformer encoder you can imagine
            self.tokenizer = AutoTokenizer.from_pretrained(hf_name)
            self.model = AutoModel.from_pretrained(hf_name)
        self.model.eval()
        # Move model to the GPU.
        if torch.cuda.is_available():
            self.model.cuda()
    
    def encode_batch(self, batch):
        """
        :param batch: list(dict('TITLE': string, 'ABSTRACT': list(string)));
            list of title and sentence tokenized abstracts
            OR list(string) in the case of embedding queries
        :return:
        """
        text_input = False if isinstance(batch[0], dict) else True
        with torch.no_grad():
            if self.model_name in {'aspire', 'mpnet1b_sc'}:
                # Passing things which aren't title, abstract for a paper will break the batcher
                assert(text_input == False)
                batch_dict = batchers.AbsSentTokBatcher.make_batch(raw_feed={'query_texts': batch},
                                                                   pt_lm_tokenizer=self.tokenizer)
                batch_reps = self.model.encode(batch_dict=batch_dict)
            else:  # Basic biencoders for which CLS is used. 'specter2_doc', 'specter2_query'
                if text_input:
                    batch_text = batch
                else:
                    batch_text = [p['TITLE'] + self.tokenizer.sep_token + ' '.join(p['ABSTRACT']) for p in batch]
                inputs = self.tokenizer(batch_text, padding=True, truncation=True,
                                        return_tensors="pt", return_token_type_ids=False, max_length=512)
                if torch.cuda.is_available():
                    for k, v in inputs.items():
                        inputs[k] = v.cuda()
                output = self.model(**inputs)
                if self.tok_pooling == 'cls':
                    embeddings = output.last_hidden_state[:, 0, :]
                elif self.tok_pooling == 'mean':
                    in_lens = torch.sum(inputs['attention_mask'], dim=1)
                    embeddings = torch.sum(output.last_hidden_state, dim=1)/in_lens.unsqueeze(1)
                else:
                    raise ValueError(f'Unknown token pooling: {self.tok_pooling}')
                embeddings = embeddings.cpu().numpy()
                batch_reps = []
                for i in range(embeddings.shape[0]):
                    batch_reps.append(embeddings[i, :][None, :])
        return batch_reps
    
    def encode(self, all_texts, batch_size=16, verbose=True):
        """
        Batch the texts passed and return reps. This can embed a list of raw strings or embed paper abstracts.
        
        :param all_texts: list(string) or dict(doc_id: dict('title': string, 'abstract': list(string)))
            a list of strings to embed or a paper abstract to embed
        :param batch_size: int;
        :param verbose: bool;
        """
        text_input = True if isinstance(all_texts, list) else False
        num_docs = len(all_texts)
        
        # Write out sentence reps incrementally.
        id2doc_reps = collections.OrderedDict()
        batch_docs = []
        batch_ids = []
        for doci, text_id in enumerate(all_texts):
            if doci % 1000 == 0 and verbose:
                print('Processing document: {:d}/{:d}'.format(doci, num_docs))
            if text_input:
                batch_docs.append(text_id)
            else:
                text_instance = all_texts[text_id]
                batch_docs.append({'TITLE': text_instance['title'],
                                   'ABSTRACT': text_instance['abstract']})
            batch_ids.append(text_id)
            if len(batch_docs) == batch_size:
                # Returns a list of matrices with sentence reps.
                batch_reps = self.encode_batch(batch_docs)
                assert (len(batch_ids) == len(batch_reps))
                for tid, doc_reps in zip(batch_ids, batch_reps):
                    assert (doc_reps.shape[1] == self.encoding_dim)
                    id2doc_reps[tid] = doc_reps
                batch_docs = []
                batch_ids = []
        # Handle left over documents.
        if len(batch_docs) > 0:
            batch_reps = self.encode_batch(batch_docs)
            assert (len(batch_ids) == len(batch_reps))
            for tid, doc_reps in zip(batch_ids, batch_reps):
                assert (doc_reps.shape[1] == self.encoding_dim)
                id2doc_reps[tid] = doc_reps
        return id2doc_reps


def test_text_embedder():
    """
    Read some abstracts and embed them.
    """
    json_path = os.path.join(os.environ['CUR_PROJ_DIR'], 'datasets_raw/cmugoldrpm')
    with codecs.open(os.path.join(json_path, f'abstracts-cmugoldrpm.json'), 'r', 'utf-8') as fp:
        pid2abstract = json.load(fp)
        print(f'Read: {fp.name}')
    print(f'Abstracts: {len(pid2abstract)}')
    
    # Embed with aspire
    # text_embedder = TextEmbedder(model_name='aspire')
    # pid2sentreps = text_embedder.encode(pid2abstract)
    # print(f'Encoded abstracts: {len(pid2sentreps)}')
    
    # Embed with specter2
    text_embedder = TextEmbedder(model_name='specter2_doc')
    pid2docreps = text_embedder.encode(pid2abstract)
    print(f'Encoded abstracts: {len(pid2docreps)}')
    
    text_embedder = TextEmbedder(model_name='specter2_query')
    title_texts = [d['title'] for pid, d in pid2abstract.items()]
    kp2reps = text_embedder.encode(title_texts)
    print(f'Encoded titles: {len(kp2reps)}')

if __name__ == '__main__':
    test_text_embedder()
