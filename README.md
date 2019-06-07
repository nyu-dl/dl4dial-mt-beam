# Multi-Turn Beam Search for Neural Dialogue Modeling

https://arxiv.org/abs/1906.00141

## Abstract

In neural dialogue modeling, a neural network is trained to predict the next utterance, and at inference time, an approximate decoding algorithm is used to generate next utterances given previous ones. While this autoregressive framework allows us to model the whole conversation during training, inference is highly suboptimal, as a wrong utterance can affect future utterances. While beam search yields better results than greedy search does, we argue that it is still greedy in the context of the entire conversation, in that it does not consider future utterances. We propose a novel approach for conversation-level inference by explicitly modeling the dialogue partner and running beam search across multiple conversation turns. Given a set of candidates for next utterance, we unroll the conversation for a number of turns and identify the candidate utterance in the initial hypothesis set that gives rise to the most likely sequence of future utterances. We empirically validate our approach by conducting human evaluation using the Persona-Chat dataset, and find that our multi-turn beam search generates significantly better dialogue responses. We propose three approximations to the partner model, and observe that more informed partner models give better performance.

![table_image](mtbeam_table.png)

## Running the code with pre-trained models

Supplementary code for submission "Multi-Turn Beam Search for Neural Dialogue Modeling"

We provide code and pretrained models for better analysis and discussion of our work.
This can be executed without GPU/CUDA with CPU-only mode. To use GPU, remove `--no-cuda` options from the scripts.  

### Install requirements

create new conda env using specification file

`conda env create -f environment.yml`

Note that each host could have some dependencies clash / mismatch. One can use this file to see what packages are required and install them individually.

Now you need to install ParlAI paths into your python env:

`cd ParlAI; python setup.py develop`

### Download pre-trained models:

`https://dl.dropboxusercontent.com/s/rh5d0xmjgjyyw22/models.zip`

Unzip models.zip as models folder in the main root folder here.
After this step there should be a folder models located in the root.

We provide pretrained models used in our paper.

### Interactive mode:

There is a script which you can use to run a real dialogue with our model.

To run using egocentric partner:

`./mtbeam_eval_interactive_egocentric.sh`

To run with mindless partner:

`./mtbeam_eval_interactive_mindless.sh`

Our model assumes that all the context is given as part of the input (check paper for details), here we include one possible
model persona which you can use (copy paste it as the first message):

`your persona: i've slightly different taste in things than most people. \n your persona: i like seafood a lot. \n your persona: i'm a natural brunette. \n your persona: i love books about science. \n your persona: i've no self control when it comes to candy. \n hello ! how are you ?`

### Validation data

There is a script to run validation epoch using pretrained model.

To run using egocentric partner:

`./mtbeam_eval_egocentric.sh`

To run using mindless partner:

`./mtbeam_eval_mindless.sh`

This code does not have transparent mode, because it was done on the MTurk side of the human evaluation. The only difference there is how human annotator persona is given to the model.


## Citation

>@misc{kulikov2019multiturn,
> 		title={Multi-Turn Beam Search for Neural Dialogue Modeling},
>    		author={Ilia Kulikov and Jason Lee and Kyunghyun Cho},
>    		year={2019},
>    		eprint={1906.00141},
>    		archivePrefix={arXiv},
>    		primaryClass={cs.CL}
>	}
