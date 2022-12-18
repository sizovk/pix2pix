# pix2pix

This repository contains implementation of pix2pix GAN model, which have been described in https://arxiv.org/abs/1611.07004

The model was trained on Edges2Handbags dataset.

### Example
![](example.jpg)

## [Training report](https://wandb.ai/k_sizov/pix2pix/reports/pix2pix-training-report--VmlldzozMTc1ODU4?accessToken=4z4arhche1oa0f9ntzrkl8qmbbaht3xokvbp76gr697lfds9jxiawvfbnxul7chl)

## Reproduce results
### Setup data
```bash
pip install -r requirements.txt
bash setup_data.sh
```

### Train model
```bash
python train.py -c configs/edges2handbags.json
python train.py -c configs/edges2shoes.json
```