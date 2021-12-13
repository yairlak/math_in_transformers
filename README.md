# Math representation in Transformer language models

# Requirements
torch
transformers (see installation options on huggingface)

# Extract perplexity for target sentences
To get perplexity for each target sentence, run: 

```get_transformer_behav.py --transformer gpt2```

With the flag --transformer you can explore different models, as long as they are recogniaziable huggingface model names. 

Make sure that the stimulus file is in the folder (or else, provide a different path using the flag --input-path):

```stimuli/``` 

This file should contain sentences in separate lines. 

The results will then be written to the folder (modifiable using the flag --output-path): 

```output/``` 

Each line contains the sentence and the corresponding perpelxity value.

