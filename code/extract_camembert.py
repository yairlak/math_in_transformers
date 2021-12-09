
import torch
from transformers import CamembertTokenizer, CamembertModel
from argparse import ArgumentParser
import numpy as np
import h5py
# from ipdb import set_trace

argp = ArgumentParser()
argp.add_argument('-i', '--input_path', default='../stimuli/test.txt')
argp.add_argument('-o', '--output_path', default='../output/test.hdf5')
args = argp.parse_args()

tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
model = CamembertModel.from_pretrained('camembert-base')

LAYER_COUNT = 12 + 1 # embedding + 12 transformer layers
FEATURE_COUNT = 768

device = "cpu"
model.to(device)
model.eval()
model.encoder.output_hidden_states = True 

with h5py.File(args.output_path, 'w') as fout:
  for index, line in enumerate(open(args.input_path)):
    line = line.strip() # Remove trailing characters
    sent_length = len(line.split())
    print(line, sent_length)
    
    tokenized_text = tokenizer.tokenize(line)
    # tokenized_text += ['▁ '] * n_blanks
    print(tokenized_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # indexed_tokens += [space_tok] * n_blanks # add the tokens separately because the tokenizer removes spaces for some reason ...
    
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    tokens_tensor.to(device)

    # token_embs = torch.zeros(LAYER_COUNT, len(tokenized_text), FEATURE_COUNT)
    with torch.no_grad():
      # for i_token in range(len(tokenized_text)):
        embs = model(tokens_tensor, output_hidden_states=True)
        # embs = embs['hidden_states'] # tupple of len(n_layers), containing tensors of shape 1*n_tokens*n_hidden
    print(embs['hidden_states'].shape)
    print(embs.keys())
        
    # tokenized to untokenized mapping
    # tok2untok = []
    # for i_tok, token in enumerate(tokenized_text): #[0:-1]): # do not consider the ending '.'
    #   if tokenized_text[i_tok][0] != '▁': # skip tokens from the last word
    #     continue

    #   # new word, get all the corresponding tokens
    #   token_ids = [i_tok] # will store all token for the current word
    #   after_tok = 1
    #   # if the sentence is not over and the next token is part of the current word
    #   while i_tok+after_tok < len(tokenized_text) \
    #     and tokenized_text[i_tok+after_tok][0] != '▁' \
    #     and tokenized_text[i_tok+after_tok][0] != '.':
    #       token_ids.append(i_tok+after_tok)
    #       after_tok += 1
    #   tok2untok.append(token_ids)

    # # # check the mapping
    # # for i in range(sent_length):
    # #   # print(line.split()[i], tokenized_text[tok2untok[i][0]], tokenized_text[tok2untok[i][-1]+1])
    # #   print(tokenized_text[tok2untok[i][0]:tok2untok[i][-1]+1])

    # token_embs = [l for l in token_embs]
    # for layer in range(LAYER_COUNT):
    #   token_embs[layer] = torch.stack([torch.mean(token_embs[layer][tok2untok[i][0]:tok2untok[i][-1]+1,:], dim=0) for i in range(sent_length)], dim=0)
    #   # token_embs[layer] = torch.stack([torch.mean(token_embs[layer][0,tok2untok[i][0]:tok2untok[i][-1]+1,:], dim=0) for i in range(sent_length)], dim=0)

    dset = fout.create_dataset(str(index), (LAYER_COUNT, sent_length, FEATURE_COUNT))
    dset[:,:,:] = np.stack([np.array(x) for x in embs])
  

