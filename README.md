# Self-supervised Label Augmentation via Input Transformations

Self-supervised learning, which learns by constructing artificial labels given only the input signals, has recently gained considerable attention for learning representations with unlabeled datasets, i.e., learning without any human-annotated supervision. In this paper, we show that such a technique can be used to significantly improve the model accuracy even under fully-labeled datasets. Our scheme trains the model to learn both original and self-supervised tasks, but is different from conventional multi-task learning frameworks that optimize the summation of their corresponding losses. Our main idea is to learn a single unified task with respect to the joint distribution of the original and self-supervised labels, i.e., we augment original labels via self-supervision of input transformation. This simple, yet effective approach allows to train models easier by relaxing a certain invariant constraint during learning the original and self-supervised tasks simultaneously. It also enables an aggregated inference which combines the predictions from different augmentations to improve the prediction accuracy. Furthermore, we propose a novel knowledge transfer technique, which we refer to as self-distillation, that has the effect of the aggregated inference in a single (faster) inference. We demonstrate the large accuracy improvement and wide applicability of our framework on various fully-supervised settings, e.g., the few-shot and imbalanced classification scenarios.

## Install dependencies

```bash
conda create -n SLA python=3.7
conda activate SLA
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install ignite -c pytorch
pip install tensorboard
```

We tested our code on the following versions:
- `pytorch==1.5.1`
- `torchvision=0.6.1`
- `ignite==0.4.0.post1`

## Training

We here provide a training script for cifar10.
```bash
python train.py \
    --dataset cifar10 --datadir data/ --batchsize 128 --num-iterations 80000 --val-freq 1000 \
    --model cresnet32 \
    --mode sla --aug rotation
```

For other training objectives, replace the `--mode` option with `baseline`, `da`, `mt`, or `sla+sd`.
For other augmentations, replace the `--aug` option with the function names in `augmentations.py`.

**Large-scale datasets.** We empirically found that summation (instead of average) of losses across self-supervised transformations could provide an accuracy gain in the large-scale datasets such as ImageNet or iNaturalist. To this end, use the `--with-large-loss` option.

## Evaluation

You can check the results in the log files stored in the `logs/` directory (`single_acc` for SLA+SI or SLA+SD; `agg_acc` for SLA+AG). To re-evaluation, use `test.py`.

