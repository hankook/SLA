# Self-supervised Label Augmentation via Input Transformations

- Authors: [Hankook Lee](https://hankook.github.io), [Sung Ju Hwang](http://www.sungjuhwang.com), [Jinwoo Shin](http://alinlab.kaist.ac.kr/shin.html) (KAIST)
- Accepted to ICML 2020

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

## BibTeX
```bib
@inproceedings{lee2020_sla,
  title={Self-supervised label augmentation via input transformations},
  author={Lee, Hankook and Hwang, Sung Ju and Shin, Jinwoo},
  booktitle={International Conference on Machine Learning},
  pages={5714--5724},
  year={2020},
  organization={PMLR}
}}
```
