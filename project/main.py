import numpy as np
import pandas as pd

np.random.seed(0)

from project.data_preprocess import bert_encode
from project.models import BERT

test_idx = pd.read_csv('data/test.csv')['id'].values
raw_data = pd.read_csv('data/lol.csv')

data_dict = {
    'id': np.arange(10876),
    'keyword': raw_data['keyword'].values,
    'location': raw_data['location'].values,
    'text': raw_data['text'].values,
    'target': raw_data['choose_one:confidence'].values
}

data = pd.DataFrame(data_dict, columns=['id', 'keyword', 'location', 'text', 'target'])
train_data = data #TODO
test_data = data #TODO

bert_model = BERT()

print('Start')
train_input = bert_encode(train_data.text.values, bert_model.bert_layer, max_len=160)
test_input = bert_encode(test_data.text.values, bert_model.bert_layer, max_len=160)
train_labels = train_data.target.values

bert_model.build_model()
print('Model build')
bert_model.summary()
print('start train')
bert_model.train(train_input, train_labels)