# GenSF: Simultaneous Adaptation of Generative Pre-trained Models andSlot Filling

This is the code for _GenSF: Simultaneous Adaptation of Generative Pre-trained Models andSlot Filling_ to be published at SIGdial 2021 (paper coming soon).

## Abstract

In transfer learning, it is imperative to achieve strong alignment between a pre-trained model and a downstream task. Prior work has done this by proposing task-specific pre-training objectives, which sacrifices the inherent scalability of the transfer learning paradigm. We instead achieve strong alignment by simultaneously modifying both the pre-trained model and the formulation of the downstream task, which is more efficient and preserves the scalability of transfer learning. We present GenSF (**Gen**erative **S**lot **F**illing), which leverages a generative pre-trained open-domain dialog model for slot filling. GenSF (1) adapts the pre-trained model by incorporating inductive biases about the task and (2) adapts the downstream task by reformulating slot filling to better leverage the pre-trained model's capabilities. GenSF achieves state-of-the-art results on two slot filling datasets with strong gains in few-shot and zero-shot settings. We achieve a **9 F1 score** improvement in zero-shot slot filling. This highlights the value of strong alignment between the pre-trained model and the downstream task. 

## Instructions for Reproducing Results

### Transformer Changes

To add a copy-mechanism to GPT-2, we edited the implementation in Transformers (version 3.0.2). Copy the files in `transformer_changes/` into `/usr/local/lib/python3.6/dist-packages/transformers/` or the relevant directory.

### Training

To train GenSF, use the following commands with the relevant data paths (all the pre-processed data is provided in this repository).

```
CUDA_VISIBLE_DEVICES=0 python3.6 train_gen.py --train_data_path restaurant8k/train_0.json --test_data_path restaurant8k/test.json --seed 42 --num_epochs 10 ;
CUDA_VISIBLE_DEVICES=0 python3.6 gen.py --train_data_path restaurant8k/train_0.json --test_data_path restaurant8k/test.json ;
```

## Zero-Shot 

To perform zero-shot experiments, comment out line 30 of `gen.py` (i.e., `model.load_state_dict(...)`) and run the following command with the relevant test data.

```
CUDA_VISIBLE_DEVICES=0 python3.6 gen.py --train_data_path restaurant8k/train_0.json --test_data_path restaurant8k/test.json ;
```

## Model Checkpoints

The model checkpoints are over 40GB in size and as such have not yet been released. In the majority of cases it would be easier to re-train the model than to download a checkpoint. However, if you would like access to a particular set of checkpoints, please contact me and I will give you access.

## Citations

If you use any of this code  please cite the following paper:

```
TBD
```

## Contact

If you have any questions about this code or the paper, please reach out to `amehri@cs.cmu.edu`.
