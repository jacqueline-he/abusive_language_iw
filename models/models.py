import numpy as np
from google.colab import drive
import pandas as pd
import matplotlib.pyplot as plt
import time
import math
import pickle
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix, \
    f1_score, precision_score, recall_score, roc_auc_score

drive.mount('/content/drive')

import tensorflow as tf

# Get the GPU device name.
device_name = tf.test.gpu_device_name()

# The device name should look like the following:
if device_name == '/device:GPU:0':
    print('Found GPU at: {}'.format(device_name))
else:
    raise SystemError('GPU device not found')

import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, AdamW

class DefaultModel(nn.Module):

  def __init__(self, bert_config, device, n_class):
    super(DefaultModel, self).__init__()
    self.n_class = n_class
    self.bert_config = bert_config
    self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)
    self.tokenizer = BertTokenizer.from_pretrained(self.bert_config)
    self.device = device

  # return: pre_softmax, torch.tensor of shape (batch_size, n_class)
  def forward(self, sents):
    sents_tensor, masks_tensor, sents_lengths = sents_to_tensor(self.tokenizer, sents, self.device)
    pre_softmax = self.bert(input_ids=sents_tensor, attention_mask=masks_tensor)
    return pre_softmax[0]

  def load(model_path: str, device):
    state_dict = torch.load('model.bin')
    print(state_dict.keys())
    model = DefaultModel('bert-base-uncased', device=device, n_class=4)
    model.load_state_dict(state_dict)
    return model

class CNNModel(nn.Module):
  def __init__(self, bert_config, device, n_class, out_channel=16):
    super(CNNModel, self).__init__()
    self.bert_config = bert_config
    self.n_class = n_class
    self.out_channel = out_channel
    self.bert = BertForSequenceClassification.from_pretrained(self.bert_config, num_labels = 4, output_hidden_states=True)
    self.out_channels = self.bert.config.num_hidden_layers*self.out_channel
    self.tokenizer=BertTokenizer.from_pretrained(self.bert_config)
    self.conv = nn.Conv2d(in_channels=self.bert.config.num_hidden_layers,
                          out_channels=self.out_channels,
                          kernel_size=(3,self.bert.config.hidden_size),
                          groups=self.bert.config.num_hidden_layers)
    self.hidden_to_softmax = nn.Linear(self.out_channels,self.n_class,bias=True)
    self.device = device

  def forward(self, sents):
    sents_tensor, masks_tensor, sents_lengths = sents_to_tensor(self.tokenizer, sents, self.device)
    outputs = self.bert(input_ids=sents_tensor,attention_mask=masks_tensor, output_hidden_states=True)
    # encoded_stack_layer = torch.stack(hidden[0:12], dim=1)
    encoded_stack_layer = torch.stack(outputs.hidden_states[0:12], dim=1)
    conv_out = self.conv(encoded_stack_layer)
    conv_out = torch.squeeze(conv_out, dim=3)
    conv_out, _ = torch.max(conv_out, dim=2)
    pre_softmax = self.hidden_to_softmax(conv_out)
    return pre_softmax

  def load(model_path: str, device):
    state_dict = torch.load('cnn-model.bin')
    print(state_dict.keys())
    model = CNNModel('bert-base-uncased', device=device, n_class=4)
    model.load_state_dict(state_dict)
    return model

def pad_sents(sents, pad_token):
  sents_padded=[]
  max_len = max(len(s) for s in sents)
  batch_size = len(sents)
  for s in sents:
    padded = [pad_token] * max_len
    padded[:len(s)] = s
    sents_padded.append(padded)

  return sents_padded

def sents_to_tensor(tokenizer, sents, device):
  tokens_list = [tokenizer.tokenize(sent) for sent in sents]
  sents_lengths = [len(tokens) for tokens in tokens_list]
  tokens_list_padded = pad_sents(tokens_list, '[PAD]')
  sents_lengths = torch.tensor(sents_lengths, device=device)

  masks=[]
  for tokens in tokens_list_padded:
    mask = [0 if token =='[PAD]' else 1 for token in tokens]
    masks.append(mask)
  masks_tensor = torch.tensor(masks, dtype=torch.long, device=device)
  tokens_id_list = [tokenizer.convert_tokens_to_ids(tokens) for tokens in tokens_list_padded]
  sents_tensor = torch.tensor(tokens_id_list, dtype=torch.long, device=device)

  return sents_tensor, masks_tensor, sents_lengths


def batch_iter(data, batch_size, shuffle=False, bert=None):
    """ Yield batches of sentences and labels reverse sorted by length (largest to smallest).
    @param data (dataframe): dataframe with ProcessedText (str) and label (int) columns
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset
    @param bert (str): whether for BERT training. Values: "large", "base", None
    """
    batch_num = math.ceil(data.shape[0] / batch_size)
    index_array = list(range(data.shape[0]))

    if shuffle:
        data = data.sample(frac=1)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]

        if bert:
            examples = data.iloc[indices].sort_values(by='ProcessedText_BERT'+bert+'_length', ascending=False)
            sents = list(examples.ProcessedText_BERT)
        else:
            examples = data.iloc[indices].sort_values(by='ProcessedText_length', ascending=False)
            sents = [text.split(' ') for text in examples.ProcessedText]

        targets = list(examples.Mapped_label.values)
        yield sents, targets


def validation(model, df_val, bert_size, loss_func, device):
  was_training = model.training
  model.eval()

  df_val = df_val.sort_values(by='ProcessedText_BERT'+bert_size+'_length', ascending=False)
  ProcessedText_BERT = list(df_val.ProcessedText_BERT)
  mapped_label = list(df_val.Mapped_label)
  val_batch_size = 32
  n_batch = int(np.ceil(df_val.shape[0]/val_batch_size))
  total_loss = 0.

  with torch.no_grad():
    for i in range(n_batch):
      sents = ProcessedText_BERT[i*val_batch_size:(i+1)*val_batch_size]
      targets = torch.tensor(mapped_label[i*val_batch_size:(i+1)*val_batch_size],
                             dtype=torch.long, device=device)
      batch_size = len(sents)
      pre_softmax = model(sents)
      batch_loss = loss_func(pre_softmax, targets)
      total_loss += batch_loss.item()*batch_size

  if was_training:
    model.train()
  return total_loss/df_val.shape[0]

path_train = '/content/drive/MyDrive/iw/data-train_NEW.csv'
  df_train = pd.read_csv(path_train, lineterminator='\n', index_col=0)
  df_train.sample(10)
  df_train = df_train.rename(columns={'Mapped_label\r':'Mapped_label'})
  df_train.columns

# USAGE:
# args is 2 parameters: 
# MODEL is in [default, cnn]
# BERT_CONFIG is in [bert-base-uncased, bert-large-uncased]
# LR is learning rate in [5e-5, 3e-5, 2e-5] (hyperparam, default)
# LR_BERT is BERT learning rate [default: 2e-5]

def train(model, bert_config, lr, lr_bert, save_path):
  label_name = ['normal', 'abusive', 'spam', 'hate speech']
  device = torch.device("cuda")
  prefix = model+'_'+bert_config
  print('prefix: ', prefix)

  start_time = time.time()
  print('Importing data...data-train.csv')
  path_train = '/content/drive/MyDrive/iw/data-train.csv'
  df_train = pd.read_csv(path_train, lineterminator='\n', index_col=0)

  print('Importing data...data-val.csv')
  path_val = '/content/drive/MyDrive/iw/data-val.csv'
  df_val = pd.read_csv(path_val, lineterminator='\n', index_col=0)


  train_label = dict(df_train.Mapped_label.value_counts())
  label_max = float(max(train_label.values()))
  train_label_weight = torch.tensor([label_max/train_label[i] for i in range(len(train_label))], device=device)
  print('Done! time elapsed %.2f sec' % (time.time() - start_time))
  print('-' * 80)

  start_time = time.time()
  print('Setting up model....')
  if model == 'default':
    model = DefaultModel(bert_config, device, len(label_name))
    optimizer = AdamW([
                {'params': model.bert.bert.parameters()}, 
                {'params': model.bert.classifier.parameters(), 'lr': float(lr)}], 
                lr=float(lr_bert), eps = 1e-8)
    # max_grad_norm=float(clip_grad)
  elif model == 'cnn':
    model = CNNModel(bert_config, device, len(label_name))
    optimizer = AdamW([
                {'params': model.bert.parameters()}, 
                {'params': model.conv.parameters(), 'lr': float(lr)},
                {'params':model.hidden_to_softmax.parameters(), 'lr':float(lr)}], 
                lr=float(lr_bert))
  else:
    print('please input a valid model')
    return

  model = model.to(device)
  print('Use device: %s' % device)
  print('Done! time elapsed %.2f sec' % (time.time() - start_time))
  print('-' * 80)


  cn_loss = torch.nn.CrossEntropyLoss(weight=train_label_weight.float(), reduction='mean')
  torch.save(cn_loss, 'loss_func') # later testing

  train_batch_size = 16 #hp 
  valid_niter = 500 #hp
  log_every = 10 #hp
  max_epoch = 5 #5
  model_save_path = save_path

  num_trial = 0
  train_iter = patience = cum_loss = report_loss = 0
  cum_examples = report_examples = epoch = 0
  hist_valid_scores = []
  train_time = begin_time = time.time()
  print('Begin Maximum Likelihood Training...')

  bert_size = bert_config.split('-')[1]

  while True:
      epoch += 1

      for sents, targets in batch_iter(df_train, batch_size = train_batch_size, shuffle=True, bert=bert_size):
        # ========================================
        #               Training
        # ========================================
        train_iter += 1
        optimizer.zero_grad()
        batch_size = len(sents)
        pre_softmax = model(sents)
        loss = cn_loss(pre_softmax, torch.tensor(targets, dtype=torch.long, device=device))
        loss.backward()
        optimizer.step()

        batch_losses_val = loss.item() * batch_size
        report_loss += batch_losses_val
        cum_loss += batch_losses_val

        report_examples += batch_size 
        cum_examples += batch_size

        if train_iter % log_every == 0:
          print('epoch %d, iter %d, avg. loss %.2f, '
                      'cum. examples %d, speed %.2f examples/sec, '
                      'time elapsed %.2f sec' % (epoch, train_iter,
                       report_loss / report_examples,
                       cum_examples,
                       report_examples / (time.time() - train_time),
                       time.time() - begin_time))
          train_time = time.time()
          report_loss = report_examples = 0

        # ========================================
        #               Validation
        # ========================================
        if train_iter % valid_niter == 0:
          print("")
          print("Running Validation...")
          print('epoch %d, iter %d, cum. loss %.2f, cum. examples %d' % (epoch, train_iter, cum_loss / cum_examples, cum_examples))
          cum_loss = cum_examples = 0.
          validation_loss = validation(model, df_val, bert_size, cn_loss, device)
          print('validation: iter %d, loss %f' % (train_iter, validation_loss))

          is_better = len(hist_valid_scores) == 0 or validation_loss < min(hist_valid_scores)
          hist_valid_scores.append(validation_loss)

          if is_better:
            patience = 0
            print('Save currently best model to [%s]' % model_save_path)
            torch.save(model.state_dict(), model_save_path)
            torch.save(optimizer.state_dict(), model_save_path + '.optim')
          elif patience < 3: # patience = how many iterations to decay LR
            patience += 1
            print('hit patience %d' % patience)

            if patience == 3: 
              num_trial += 1
              print('hit #%d trial' % num_trial)
              if num_trial == 3:
                print('early stop!')
                break
            
              # decay lr, and restore from previously best checkpoint
              print('load previously best model and decay learning rate to %f%%' %
                               (0.5)*100)
            
              # load model
              state_dict = torch.load('model.bin')
              model.load_state_dict(state_dict)
              model = model.to(device)

              print('restore parameters of the optimizers')
              optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

              # set new lr
              optimizer.lr *= 0.5

      if epoch == int(max_epoch):
        print('Reached max. number of epochs!')
        break

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, path='cm', cmap=plt.cm.Reds):
  if not title:
    if normalize:
      title = 'Normalized Confusion Matrix'
    else:
      title = 'Confusion matrix, without normalization'
  cm = confusion_matrix(y_true, y_pred)

  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  pickle.dump(cm, open(path, 'wb'))

  fig, ax = plt.subplots()
  im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
  ax.figure.colorbar(im, ax=ax)
  # We want to show all ticks...
  ax.set(xticks=np.arange(cm.shape[1]),
         yticks=np.arange(cm.shape[0]),
         # ... and label them with the respective list entries
         xticklabels=classes, yticklabels=classes,
         title=title,
         ylabel='True label',
         xlabel='Predicted label')

  # Rotate the tick labels and set their alignment.
  plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
          rotation_mode="anchor")

  # Loop over data dimensions and create text annotations.
  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
      ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
  fig.tight_layout()
  return ax

def test(model_name, bert_config):
  label_name = ['normal', 'abusive', 'spam', 'hate speech']
  device = torch.device("cuda")
  prefix = model_name+'_'+bert_config
  print('prefix: ', prefix)

  print('Load best model...')
  if model_name == 'default':
    model = DefaultModel.load('/content/model.bin', device)
  elif model_name == 'cnn':
    model = CNNModel.load('/content/cnn-model.bin', device)
  model.to(device)
  model.eval()

  path_test = '/content/drive/MyDrive/iw/data-test.csv'
  df_test = pd.read_csv(path_test, lineterminator='\n', index_col=0)
  test_batch_size = 32
  n_batch = int(np.ceil(df_test.shape[0]/test_batch_size))
  cn_loss = torch.load('loss_func').to(device)

  ProcessedText_BERT = list(df_test.ProcessedText_BERT)
  mapped_label = list(df_test.Mapped_label)
  test_loss = 0.
  prediction = []
  prob = []
  softmax = torch.nn.Softmax(dim=1)
  total = 0
  with torch.no_grad():
    for i in range(n_batch):
      sents = ProcessedText_BERT[i*test_batch_size: (i + 1) * test_batch_size]
      targets = torch.tensor(mapped_label[i*test_batch_size: (i + 1) * test_batch_size],
                             dtype=torch.long, device=device)
      batch_size = len(sents)
      pre_softmax = model(sents)
      batch_loss = cn_loss(pre_softmax, targets)
      test_loss += batch_loss.item()*batch_size
      prob_batch = softmax(pre_softmax)
      prob.append(prob_batch)
      total += batch_size

      prediction.extend([t.item() for t in list(torch.argmax(prob_batch, dim=1))])

  prob = torch.cat(tuple(prob), dim=0)
  print("test loss: ", test_loss)
  print("num ", df_test.shape[0])
  loss = test_loss/df_test.shape[0]

  pickle.dump([label_name[i] for i in prediction], open(prefix+'_test_prediction', 'wb'))
  pickle.dump(prob.data.cpu().numpy(), open(prefix + '_test_prediction_prob', 'wb'))

  accuracy = accuracy_score(df_test.Mapped_label.values, prediction)
  matthews = matthews_corrcoef(df_test.Mapped_label.values, prediction)

  precisions = {}
  recalls = {}
  f1s = {}
  aucrocs = {}
  total_prediction = []
  total_recall = []
  total_f1 = []


  print("total f1: ", f1_score(prediction, df_test.Mapped_label.values, average='weighted'))
  print("total precision: ", precision_score(prediction, df_test.Mapped_label.values, average='weighted'))
  print("total recall: ", recall_score(prediction, df_test.Mapped_label.values, average='weighted'))

  for i in range(len(label_name)):
    prediction_ = [1 if pred == i else 0 for pred in prediction]
    true_ = [1 if label == i else 0 for label in df_test.Mapped_label.values]
    f1s.update({label_name[i]: f1_score(true_, prediction_)})
    precisions.update({label_name[i]: precision_score(true_, prediction_)})
    recalls.update({label_name[i]: recall_score(true_, prediction_)})
    aucrocs.update({label_name[i]: roc_auc_score(true_, list(t.item() for t in prob[:, i]))})

  metrics_dict = {'loss': loss, 'accuracy': accuracy, 'matthews coef': matthews, 'precision': precisions,
                         'recall': recalls, 'f1': f1s, 'aucroc': aucrocs}

  pickle.dump(metrics_dict, open(prefix+'_evaluation_metrics', 'wb'))

  cm = plot_confusion_matrix(list(df_test.Mapped_label.values), prediction, label_name, normalize=False,
                           path=prefix+'_test_confusion_matrix', title='confusion matrix for test dataset')
  plt.savefig(prefix+'_test_confusion_matrix', format='png')
  cm_norm = plot_confusion_matrix(list(df_test.Mapped_label.values), prediction, label_name, normalize=True,
                           path=prefix+'_test normalized_confusion_matrix', title='normalized confusion matrix for test dataset')
  plt.savefig(prefix+'_test_normalized_confusion_matrix', format='png')

  print('TESTING RESULTS')
  print('-' * 80)
  print('Loss: %.3f' % loss)
  print('Accuracy: %.3f' % accuracy)
  print('Matthews Coefficient: %.3f' % matthews)
  print('-' * 80)
  for i in range(len(label_name)):
    print('Precision score for %s: %.3f' % (label_name[i], precisions[label_name[i]]))
    print('Recall score for %s: %.3f' % (label_name[i], recalls[label_name[i]]))
    print('F1 score for %s: %.3f' % (label_name[i], f1s[label_name[i]]))
    print('AUC ROC score for %s: %.3f' % (label_name[i], aucrocs[label_name[i]]))
    print('-' * 80)



#TRAIN AND TEST DEFAULTMODEL
train('default', 'bert-base-uncased', .001, 2e-5, 'model.bin')
test('default', 'bert-base-uncased')


#SAVE DEFAULTMODEL
device = torch.device("cuda")
model = DefaultModel.load('/content/model.bin', device)
model.to(device)
model_save_name = 'default-model.bin'
path = F"/content/drive/My Drive/COS397 IW/{model_save_name}"
torch.save(model.state_dict(), path)


#TRAIN AND TEST CNNMODEL
train('cnn', 'bert-base-uncased', .001, 2e-5, 'cnn-model.bin')
test('cnn', 'bert-base-uncased')

#SAVE CNNMODEL
device = torch.device("cuda")
model = DefaultModel.load('/content/model.bin', device)
model.to(device)
model_save_name = 'cnn-model.bin'
path = F"/content/drive/My Drive/COS397 IW/{model_save_name}"
torch.save(model.state_dict(), path)

