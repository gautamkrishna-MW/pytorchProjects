
import torch
import os
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

class Dictionary():
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1
            
    def __len__(self):
        return len(self.word2idx)

class newDictionary():
    def __init__(self):
        self.setofWords = set()
        self.word2idx = {}
        self.idx2word = {}

    def add_word(self, word):
        self.setofWords.add(word)

    def generateDict(self):
        num_words = len(self.setofWords)
        idx = [i for i in range(num_words)]
        self.word2idx = dict(zip(self.setofWords,idx))
        self.idx2word = dict(zip(idx,self.setofWords))
        
    def __len__(self):
        return len(self.word2idx)

class Corpus():
    
    def __init__(self):
        self.dictionary = newDictionary()

    def get_data(self, path, batch_size=20):
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words: 
                    self.dictionary.add_word(word)
        
        self.dictionary.generateDict()

        #Create a 1-D tensor which contains index of all the words in the file with the help of word2idx
        ids = torch.LongTensor(tokens)
        token = 0
        with open(path, 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids[token] = self.dictionary.word2idx[word]
                    token += 1
        # no of required batches            
        num_batches = ids.shape[0] // batch_size     
        #Remove the remainder from the last batch , so that always batch size is constant
        ids = ids[:num_batches*batch_size]
        # return (batch_size,num_batches)
        ids = ids.view(batch_size, -1)
        return ids

class LSTM(nn.Module):
    
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size) # maps words to feature vectors
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True) # LSTM layer
        self.linear = nn.Linear(hidden_size, vocab_size) # Fully connected layer

    def forward(self, x, h):
        # Perform Word Embedding 
        x = self.embed(x)

        out, (h, c) = self.lstm(x, h) # (input , hidden state)
        
        # Reshape output to (batch_size*sequence_length, hidden_size)
        out = out.reshape(out.size(0)*out.size(1), out.size(2))
        
        # Decode hidden states of all time steps
        out = self.linear(out)
        return out, (h, c)

# to Detach the Hidden and Cell states from previous history
def detach(states):
    return [state.detach() for state in states]

def mainFunc():
    
    embed_size = 128    # Embedding layer size , input to the LSTM
    hidden_size = 512  # Hidden size of LSTM units
    num_layers = 1      # no LSTMs stacked
    num_epochs = 10     # total no of epochs
    batch_size = 50     # batch size
    seq_length = 100     # sequence length
    learning_rate = 0.0005 # learning rate

    corpObj = Corpus()
    ids = corpObj.get_data('train.txt', batch_size=batch_size)
    vocab_size = len(corpObj.dictionary)

    num_batches = ids.shape[1] // seq_length
    model = LSTM(vocab_size, embed_size, hidden_size, num_layers)
    model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        # initial hidden and cell states
        states = (torch.zeros(num_layers, batch_size, hidden_size,device=torch.device('cuda')),
                  torch.zeros(num_layers, batch_size, hidden_size,device=torch.device('cuda')))
    
        for i in range(0, ids.size(1) - seq_length, seq_length):
        
            #move with seq length from the the starting index and move till - (ids.size(1) - seq_length)
        
            # prepare mini-batch inputs and targets
            inputs = ids[:, i:i+seq_length].cuda() # fetch words for one seq length  
            targets = ids[:, (i+1):(i+1)+seq_length].cuda() # shifted by one word from inputs
        
            states = detach(states)

            states[0].to(torch.device('cuda'))
            states[1].to(torch.device('cuda'))
            outputs,states = model(inputs.to(torch.device('cuda')), states)
            loss = criterion(outputs, targets.reshape(-1))

            model.zero_grad()
            loss.backward()
         
            #The gradients are clipped in the range [-clip_value, clip_value]. This is to prevent the exploding gradient problem
            clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
              
            step = (i+1) // seq_length
            if step % 100 == 0:
                print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))


    # Test the model
    with torch.no_grad():
        outfilename = 'results3.txt'
        with open(outfilename, 'w') as f:
            #intial hidden ane cell states
            state = (torch.zeros(num_layers, 1, hidden_size).cuda(),
                     torch.zeros(num_layers, 1, hidden_size).cuda())
        
            # Select one word id randomly and convert it to shape (1,1)
            input = torch.randint(0,vocab_size, (1,)).long().unsqueeze(1).cuda()
                                # (min , max , shape) , convert to long tensor and make it a shape of 1,1 

            for i in range(500):
                output, _ = model(input, state)

            
                # Sample a word id from the exponential of the output 
                prob = output.exp()
                word_id = torch.multinomial(prob, num_samples=1).item()
                #print(word_id)

            
                # Replace the input with sampled word id for the next time step
                input.fill_(word_id)

                # Write the results to file
                word = corpObj.dictionary.idx2word[word_id]
                word = '\n' if word == '<eos>' else word + ' '
                f.write(word)

            
                if (i+1) % 100 == 0:
                    print('Sampled [{}/{}] words and save to {}'.format(i+1, 500, outfilename))

if __name__=='__main__':
    mainFunc()