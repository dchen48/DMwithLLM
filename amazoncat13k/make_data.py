import gzip
import json
import pandas as pd
import torch

def load_gz_json(file_path):
    data = []
    with gzip.open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.decode('utf-8')))
    return data
    
def categoryCount(labels, ids):
    d = {}
    for label_ids in ids:
        for label_id in label_ids:
            label = labels[label_id]
            if label not in d:
                d[label] = 1
            else:
                d[label] += 1
    return d

def getCategories(threshold, labels, ids):
    from sentence_transformers import SentenceTransformer
    import gzip
    import json
    import re
    
    model = SentenceTransformer('bert-base-nli-mean-tokens')
        
    for label, freq in categoryCount(labels, ids).items():
        if freq >= threshold:
            embedcat = model.encode([label])[0]
            yield label, embedcat

def datasetStats(threshold, labels, ids):
    numclasses = len([ entity for entity, freq in categoryCount(labels, ids).items() if freq >= threshold ])
    return { 'numclasses': numclasses, 'numexamples': threshold * numclasses }

def makeData(threshold, categories, titles, contents, ids, labels):
    from collections import defaultdict
    from sentence_transformers import SentenceTransformer
    import json
    
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    catcount = defaultdict(int)
    batchencode, batchids = [], []
    for title, content, target_ids in zip(titles, contents, ids):
        flag = False
      
        if True:
            batchencode.append(title)
            batchencode.append(content)
            stored_ids = []
            for id in target_ids:
                label = labels[id]
                if label in categories:
                    stored_ids.append(id)
            batchids.append(stored_ids)

            if len(batchids) == 5:
                embed = model.encode(batchencode)
                for n, current_ids in enumerate(batchids):
                    embedtitle, embedcontent = embed[2*n], embed[2*n+1]
                    #entityord, entityvec = categories[entity]
                    yield {
                            'ids': current_ids,
                            'title': embedtitle,
                            'content': embedcontent,
                            'text_title':  batchencode[2*n],
                            'text_content': batchencode[2*n+1]}

                batchencode, batchids = [], []
                
        if len(batchencode):
            embed = model.encode(batchencode)

            for n, current_ids in enumerate(batchids):
                    embedtitle, embedcontent = embed[2*n], embed[2*n+1]
                    #entityord, entityvec = categories[entity]
                    yield {
                            'ids': current_ids,
                            'title': embedtitle,
                            'content': embedcontent,
                            'text_title':  batchencode[2*n],
                            'text_content': batchencode[2*n+1] }
            batchencode, batchids = [], []

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, threshold, titles, contents, ids, labels):
        from tqdm.notebook import tqdm
        self.labelfeats = { k: (n, v) for n, (k, v) in enumerate(getCategories(threshold, labels, ids)) }
        Xs = []
        ys = []
        text_titles = []
        text_contents = []
        for n, what in tqdm(enumerate(makeData(threshold, self.labelfeats, titles, contents, ids, labels))):
            title = torch.tensor(what['title'])
            content = torch.tensor(what['content'])
            text_title = what['text_title']
            text_content = what['text_content']
            
            text_titles.append(text_title)
            text_contents.append(text_content)
                        
            Xs.append(torch.cat((title, content)).unsqueeze(0))
            cur_ids = what['ids']
            y = []
            for id in cur_ids:
                label = labels[id]
                y.append(self.labelfeats[label][0])
            ys.append(y)
            
        self.Xs = torch.cat(Xs, dim=0)
        self.text_titles = text_titles
        self.text_contents = text_contents
        self.ys = ys
            
    def __len__(self):
        return self.Xs.shape[0]

    def __getitem__(self, index):
        return self.Xs[index], self.text_titles[index], self.text_contents[index], self.ys[index]

def makeMyDataset(threshold, titles, contents, ids, labels):
    import gzip
    
    foo = MyDataset(threshold, titles, contents, ids, labels)
    with gzip.open(f'mydataset.{threshold}.pickle.gz', 'wb') as handle:
        import pickle
        pickle.dump(foo, handle, protocol=pickle.HIGHEST_PROTOCOL)

def loadMyDataset(threshold):
    import gzip
    
    with gzip.open(f'mydataset.{threshold}.pickle.gz', 'rb') as handle:
        import pickle
        return pickle.load(handle)
        

with open('AmazonCat-13K/Yf.txt', 'r', encoding='ISO-8859-1') as f:
    labels = f.readlines()
labels = [label.strip() for label in labels]

trn_data = load_gz_json('AmazonCat-13K/trn.json.gz')
trn_df = pd.DataFrame(trn_data)

titles = trn_df['title']
contents = trn_df['content']
ids = trn_df['target_ind']

threshold = 0
makeMyDataset(threshold, titles, contents, ids, labels)
