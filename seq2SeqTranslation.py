
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os , string
# we will be using pytorch to build the model , hence importing required torch essentials
import torch
from torch.utils.data import DataLoader , TensorDataset
from torch import mean, nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from network import EncoderRNN, DecoderRNN

import kagglehub
path = kagglehub.dataset_download("msarmi9/englishkorean-multitarget-ted-talks-task-mttt")
print(path)

# file paths
pathhead = "C:/Users/gauta/.cache/kagglehub/datasets/msarmi9/englishkorean-multitarget-ted-talks-task-mttt/versions/1/multitarget-ted/en-ko/raw"
eng_train = pathhead + '/ted_train_en-ko.raw.en'
ko_train = pathhead + '/ted_train_en-ko.raw.ko'

eng_test = pathhead + '/ted_test1_en-ko.raw.en'
ko_test = pathhead + '/ted_test1_en-ko.raw.ko'


# function to read raw text file
def read_text(filename):
    # open the file
    file = open(filename, mode='rt', encoding='utf-8')
    # read all text
    text = file.readlines()
    file.close()
    return text

# reading the files
df_eng_train = read_text(eng_train)
df_ko_train = read_text(ko_train)

df_eng_test = read_text(eng_test)
df_ko_test = read_text(ko_test)

# looking at the length of the different datasets
print(len(df_eng_train) , len(df_ko_train) , len(df_eng_test) , len(df_ko_test))

# Data pre-processing Remove punctuation
df_eng_train = [s.translate(str.maketrans('', '', string.punctuation)) for s in df_eng_train]
df_ko_train = [s.translate(str.maketrans('', '', string.punctuation)) for s in df_ko_train]
df_eng_test = [s.translate(str.maketrans('', '', string.punctuation)) for s in df_eng_test]
df_ko_test = [s.translate(str.maketrans('', '', string.punctuation)) for s in df_ko_test]

# looking into 5 pair of sentence from both the languages
for i in range(5):
    print("English: {} \n Korean: {} \n".format(df_eng_train[i].strip(), df_ko_train[i].strip()))

# to lower the words and remove new lines charecters
eng_word_list_train = []
eng_word_list_test = []
kor_word_list_train = []
kor_word_list_test = []
eng_word_list = []
kor_word_list = []
for i in range(len(df_eng_train)):
    eng_word_list_train += df_eng_train[i].lower().rstrip("\n").split()
    kor_word_list_train += df_ko_train[i].lower().rstrip("\n").split()
    
for i in range(len(df_eng_test)):
    eng_word_list_test += df_eng_test[i].lower().rstrip("\n").split()
    kor_word_list_test += df_ko_test[i].lower().rstrip("\n").split()
    
eng_word_list = eng_word_list_train + eng_word_list_test
kor_word_list = kor_word_list_train + kor_word_list_test

# bulding the vocabulary for English and Korean Language
# first we will build the index 2 word mapping
en_index2word = ["<PAD>", "<SOS>", "<EOS>"] # <PAD> will be the zero'th index
ko_index2word = ["<PAD>", "<SOS>", "<EOS>"]

en_index2word = en_index2word + list(set(eng_word_list))
ko_index2word = ko_index2word + list(set(kor_word_list))
print("Index to word, Done")
print(len(en_index2word),len(ko_index2word))

# word to index
en_word2index = {token:idx for idx,token in enumerate(en_index2word)}
ko_word2index = {token:idx for idx,token in enumerate(ko_index2word)}

def encoding_padding(vocab, sentence, seq_length):
    
    SOS = [vocab["<SOS>"]] # we will add start of sentence and end of sentence token at each sentence
    EOS = [vocab["<EOS>"]]
    PAD = [vocab["<PAD>"]]

    sentence = sentence.lower().split()  
    if len(sentence) < (seq_length - 2): # -2 is for SOS and EOS
        pads = ((seq_length - 2) - len(sentence)) * PAD
        encoding = [vocab[word] for word in sentence]
        return SOS + encoding + EOS + pads
    else:
        encoding = [vocab[word] for word in sentence[:(seq_length - 2)]]
        return SOS + encoding + EOS


# Truncating sentences to a fixed-length
seq_length = 25 # max(df_eng_train,key=len)

# encoding every sentence in train and test
encoded_train_en = [encoding_padding(en_word2index,sent,seq_length) for sent in df_eng_train]
encoded_train_ko = [encoding_padding(ko_word2index,sent,seq_length) for sent in df_ko_train]
encoded_test_en = [encoding_padding(en_word2index,sent,seq_length) for sent in df_eng_test]
encoded_test_ko = [encoding_padding(ko_word2index,sent,seq_length) for sent in df_ko_test]

# creating numpy array for train and test
train_x = np.array(encoded_train_en)
train_y = np.array(encoded_train_ko)

test_x = np.array(encoded_test_en)
test_y = np.array(encoded_test_ko)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Creating the Torch tensor dataloaders
train_data = TensorDataset(torch.from_numpy(train_x) , torch.from_numpy(train_y))
test_data = TensorDataset(torch.from_numpy(test_x) , torch.from_numpy(test_y))

batch_size = 10

train_dl = DataLoader(train_data,shuffle=False,batch_size=batch_size,drop_last=True)
test_dl = DataLoader(test_data,shuffle=False,batch_size=batch_size,drop_last=True)

#initializing the networks
hidden_size = 256
encoder = EncoderRNN(len(en_index2word), hidden_size).to(device)
decoder = DecoderRNN(hidden_size, len(ko_index2word)).to(device)

# looking at the networks
print(encoder)
print(decoder)

criterion = nn.CrossEntropyLoss().cuda() # loss function
enc_optimizer = torch.optim.Adam(encoder.parameters(), lr = 3e-2) #encoder optimizer with models params and Learning rate
dec_optimizer = torch.optim.Adam(decoder.parameters(), lr = 3e-2) #decoder optimizer with models params and Learning rate

input_length = target_length = seq_length # setting the sizes
losses = []
SOS = en_word2index["<SOS>"]
EOS = en_word2index["<EOS>"]


epochs = 20

for epoch in range(epochs):
    print('Epoch:',epoch)
    for idx, batch in enumerate(train_dl):

        # Creating initial hidden states for the encoder
        encoder_hidden = encoder.initHidden(batch_size)

        # Sending to device 
        encoder_hidden = encoder_hidden.to(device)

        # Assigning the input and sending to device
        input_tensor = batch[0].to(device)

        # Assigning the output and sending to device
        target_tensor = batch[1].to(device)
        

        # Clearing gradients
        enc_optimizer.zero_grad()
        dec_optimizer.zero_grad()

        # Enabling gradient calculation
        with torch.set_grad_enabled(True):
            
            # Feeding batch into encoder
            encoder_output, encoder_hidden = encoder(input_tensor, encoder_hidden)

            # This is a placeholder tensor for decoder outputs. We send it to device as well
            dec_result = torch.zeros(target_length, batch_size, len(ko_index2word)).to(device)

            # Creating a batch of SOS tokens which will all be fed to the decoder
            decoder_input = target_tensor[:, 0].unsqueeze(dim=0).to(device)

            # Creating initial hidden states of the decoder by copying encoder hidden states
            decoder_hidden = encoder_hidden

            # For each time-step in decoding:
            for i in range(1, target_length):
                
                # Feed input and previous hidden states 
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                
                # Finding the best scoring word
                best = decoder_output.argmax(1) 

                # Assigning next input as current best word
                decoder_input = best.unsqueeze(dim=0) 

                # Creating an entry in the placeholder output tensor
                dec_result[i] = decoder_output


            # Creating scores and targets for loss calculation
            scores = dec_result.transpose(1, 0)[1:].reshape(-1, dec_result.shape[2])
            if batch_size > 1:
                targets = target_tensor[1:].reshape(-1)
            else:
                targets = target_tensor
            targets_long = targets.to(torch.long)

            # Calculating loss
            loss = criterion(scores, targets_long)
            
            # Performing backprop and clipping excess gradients
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1)

            enc_optimizer.step() 
            dec_optimizer.step()

            # Keeping track of loss
            losses.append(loss.item())
            # printing loss for every 100 iteration
            if idx % 100 == 0:
                print(idx, sum(losses)/len(losses))