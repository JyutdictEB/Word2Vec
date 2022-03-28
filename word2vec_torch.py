from collections import namedtuple
import extractor
import numpy as np
import torch 
from torch.autograd import Variable
import torch.functional as F
import torch.nn.functional as F

data_path = 'raw.txt'
cpu = torch.device('cpu')

def generate_index(text):
  print("generating index")
  Word2idx = namedtuple("Word2idx",['word2idx','idx2word','sentences'])
  vocab = []
  sentence = []
  for line in text:
    tokenized_sentence = extractor.extract_gram(line,1)
    sentence.append(list(filter(lambda w: w not in extractor.erase_symbol,tokenized_sentence)))
    for each in tokenized_sentence:
      if each not in vocab and each not in extractor.erase_symbol:
        vocab.append(each)
  w2i = {w: idx for (idx,w) in enumerate(vocab)}
  i2w = {idx: w for (idx,w) in enumerate(vocab)}
  return Word2idx(w2i, i2w, sentence)

def loadStream(path):
  with open(path,mode='r',encoding='utf-8') as f:
    while f.readable():
      l = f.readline()
      if len(l) == 0:
        return
      yield str(l).strip()
def index_center_context_word(window_size:int,word2index,tokenized_sentence):
  idx_pairs = []
  for sentence in tokenized_sentence:
    indices = [word2index[word] for word in sentence] # of a sentence
    for center_word_pos in range(len(indices)):
      for cur_w in range(-window_size,window_size + 1):
        context_word_pos = center_word_pos + cur_w
        if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
            continue
        context_word_idx = indices[context_word_pos]
        idx_pairs.append((indices[center_word_pos], context_word_idx))
  return np.array(idx_pairs)


def get_input_layer(word_idx,vocab_size):
    x = torch.zeros(vocab_size).float()
    x[word_idx] = 1.0
    return x


idxs = generate_index(loadStream(data_path))
idx_pairs = index_center_context_word(1,idxs.word2idx,idxs.sentences)
vocab_size = len(idxs.word2idx)

embedding_dims = 5
W1 = Variable(torch.randn(embedding_dims, vocab_size).float(), requires_grad=True)
W2 = Variable(torch.randn(vocab_size, embedding_dims).float(), requires_grad=True)
num_epochs = 101
learning_rate = 0.001
print('ready to train')
#train
for epo in range(num_epochs):
  loss_val = 0
  for data, target in idx_pairs:
      x = Variable(get_input_layer(data,vocab_size)).float()
      y_true = Variable(torch.from_numpy(np.array([target])).long())

      z1 = torch.matmul(W1, x)
      z2 = torch.matmul(W2, z1)
  
      log_softmax = F.log_softmax(z2, dim=0)

      loss = F.nll_loss(log_softmax.view(1,-1), y_true)
      loss_val += loss.data
      loss.backward()
      W1.data -= learning_rate * W1.grad.data
      W2.data -= learning_rate * W2.grad.data

      W1.grad.data.zero_()
      W2.grad.data.zero_()

  print(f'Loss at epo {epo}: {loss_val/len(idx_pairs)}')
