import codecs
import json
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

pid2abstract = {}
with codecs.open('abstracts-csfcube-preds.jsonl', 'r', 'utf-8') as absfile:
    for line in absfile:
        injson = json.loads(line.strip())
        pid2abstract[injson['paper_id']] = injson

class Model:
    def __init__(self, model):
        # load model and tokenizer
        if model == "specter":
            self.tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
            self.model = AutoModel.from_pretrained('allenai/specter')
        elif model == "scincl":
            self.tokenizer = AutoTokenizer.from_pretrained('malteos/scincl')
            self.model = AutoModel.from_pretrained('malteos/scincl')
        self.doc_encoder = AutoModel.from_pretrained('allenai/specter')
        self.doc_encoder.eval()
        if torch.cuda.is_available():
            self.doc_encoder.cuda()

    def embed(self, batch_text):
        """
        Do a forward pass through the encoder and get embeddings.
        """
        print(batch_text)
        inputs = self.tokenizer(batch_text, padding=True, truncation=True,
                           return_tensors="pt", return_token_type_ids=False, max_length=512)
        # Move the data to the GPU.
        if torch.cuda.is_available():
           for k, v in inputs.items():
              inputs[k] = v.cuda()
        # This makes the model consume lesser memory
        with torch.no_grad():
            output = self.model(**inputs)
        embeddings = output.last_hidden_state[:, 0, :]
        return embeddings.cpu().numpy()
    
    def embed_text(self, doc_text, batch_size=16):
        """
        Embed and return the docs.
        doc_text: list(string); the entire list of documents we want to embed.
        """
        # Embed batch_docs at a time with the model. batch_docs will have batch_size number of texts
        batch_docs = []
        batch_start_idx = 0
        num_docs = len(doc_text)
        doc_embeddings = np.empty((num_docs, 768))  # storage for the whole set of embeddings
        for doci, abs_text in enumerate(doc_text):
            if doci % 100 == 0:
                print('Processing document: {:d}/{:d}'.format(doci, num_docs))
            batch_docs.append(abs_text)
            if len(batch_docs) == batch_size:
                batch_reps = self.embed(batch_docs)
                batch_docs = []
                doc_embeddings[batch_start_idx:batch_start_idx+batch_size, :] = batch_reps
                batch_start_idx = batch_start_idx+batch_size
            # Handle left over sentences in doc_text.
            if len(batch_docs) > 0:
                batch_reps = self.embed(batch_docs)
                final_bsize = batch_reps.shape[0]
                doc_embeddings[batch_start_idx:batch_start_idx + final_bsize, :] = batch_reps
        return doc_embeddings
    
    def execute(self):
        with open('abstracts-preds.json', 'r') as file:
            data = json.load(file)
            int_idx2pid = {}
            # embed all the documents
            doc_text = []
            i = 0
            for key, value in data.items():
                json_data = data[key]
                paper_id = json_data['paper_id']
                title = json_data['title']
                abstract = "".join(json_data['abstract'])
                cur_doc = f"{title} {self.tokenizer.sep_token} {abstract}"
                doc_text.append(cur_doc)
                int_idx2pid[i] = paper_id
                i+= 1
                # break at 10k documents
                if i == 500:
                    break
            doc_embeds = self.embed_text(doc_text)

            # initialize a nearest neighbor data structure for retriving 200 documents with L2 distance
            document_neighbor_index = NearestNeighbors(n_neighbors=200, metric='minkowski', p=2)
            document_neighbor_index.fit(doc_embeds)

            query_text = []
            query_ids = []
            # choose facet
            facet = "background"
            fileName = f"gold/test-pid2anns-csfcube-{facet}.json"
            with codecs.open(fileName, 'r', 'utf-8') as fp:
                qpid2pool = json.load(fp)
                for qpid in qpid2pool.keys():
                    cur_query = pid2abstract[qpid]['title'] + " " + self.tokenizer.sep_token + " ".join(pid2abstract[qpid]['abstract'])
                    query_text.append(cur_query)
                    query_ids.append(qpid)
            # now embed the queries
            query_embeds = self.embed_text(query_text)

            # retrieve the nearest neighbors for all queries at once (sklearn implements batching internally)
            # neighbor_distances will be shape: num_queries x 200
            # neighbor_indexes: of shape num_queries x 200, will be the integer index of the neighbor document in doc_embeds. You can retrieve the pid using int_idx2pid
            neighbor_distances, neighbor_indexes = document_neighbor_index.kneighbors(query_embeds)
            return neighbor_distances, neighbor_indexes, query_ids, int_idx2pid

resJSON = {}
model = Model("specter")
neighbor_distances, neighbor_indexes, query_ids, int_idx2pid = model.execute()
for i in range(len(query_ids)):
    neighbor_indexes[i] = [int_idx2pid[elem] for elem in neighbor_indexes[i]]
    IndexAndDist = list(zip(neighbor_indexes[i], neighbor_distances[i]))
    ranked_pool = list(sorted(IndexAndDist, key=lambda x: x[1]))
    resJSON[query_ids[i]] = ranked_pool

resJSON_serializable = {
    key: [(int(cpid), float(dist)) for cpid, dist in value]
    for key, value in resJSON.items()
}

outputFileName = f"800kRanked.json"
with open(outputFileName, 'w') as json_file:
    json.dump(resJSON_serializable, json_file)