# mt-beam-parlai
Parlai external agent for Multi-turn beam search

Do the following steps to use this repository:

```bash
git clone git@github.com:facebookresearch/ParlAI.git
git clone git@github.com:uralik/mt-beam-parlai ParlAI/parlai_external
cd ParlAI; python setup.py develop
```

Now you can do `from parlai_external.X.Y import Z` to use the modules in this project.

## TODO
[x] self.partner_model creation, save, load
[x] Convai2lookaheadTeacher agent provides immediate future text and labels (prolly useful for some metrics later)
[] Add self.partner_model to optimizer
[] Extend train_step to train self.partner_model on the input where personas context is removed

## Training
There are 2 files in scripts: train.py is just wrapper over default train_model.py script in parlai (but we can extend it),
run_train.sh specifies all necessary parameters to start from for now.

Notice --dict-file there. To get the dictionary, run this:
`python ParlAI/parlai/scripts/build_dict.py -t convai2:self --dict-file mtbeam-data/convai2.dict --dict-include-valid True`
