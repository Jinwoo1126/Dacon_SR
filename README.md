# **Dacon AISR**

![Colab](https://img.shields.io/static/v1?label=Demo&message=Colab&color=orange)

**🔥 Team : 모두연 포레버** <br>
**🔥 Member : 박수철, 장진우, 윤성국, 양성모**

<br>

## **Descriptions**
---

<br>

**🚩 Score**
---
>  **Private Score** <p>
▶️ 25.00327 **[1st]** <p>
**Public Score** <p>
▶️ 25.96657 **[2nd]**

<br>

**💻 Environment**
---

```
<Main>
OS : 
CPU : 
GPU : A100 * 8 
Memory : 

&

<Sub>
Colab Pro+ 
```


<br>

## **Competition strategies**
---
**1. Data Augmentation**

 - Rotation

 - Flip

**2. Post Processing**

 - Geometric Self-Ensemble [[https://arxiv.org/pdf/1707.02921.pdf](https://arxiv.org/pdf/1707.02921.pdf)]

**3. Loss function**

 - MSELoss ( PSNR socre is directly related with `MSE` )

**4. Increase `num_feat = 64 -> 96`** 

**5. gt_size increase `gt_size = 64 -> 512`**

**6. Multistep LR optimizer**
 - Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones of 200,000
 - In the second half of the learning, detailed learning is performed.

<br>

**Main configuration & Hyper parameters**
```
manuel_seed : 0
gt_size : 512
num_feat : 96

train:
  ema_decay: 0.999
  optim_g:
    type: Adam
    lr: !!float 1e-4
    weight_decay: 0
    betas: [0.9, 0.99]

pixel_opt:
  type: MSELoss
  loss_weight: 1.0
  reduction: mean
```

**Configuration in `./options/finetune_realesrnet_x4plus_pairdata.yml`**
```yaml
# general settings
name: finetune_RealESRNetx4plus_400k_pairdata
model_type: RealESRNetModel
scale: 4
num_gpu: auto
manual_seed: 0

# USM the ground-truth
l1_gt_usm: True
percep_gt_usm: True
gan_gt_usm: False

high_order_degradation: False # do not use the high-order degradation generation process

# dataset and data loader settings
datasets:

  train: # the 1st test dataset
    name: Dacon_train
    type: PairedImageDataset
    dataroot_gt: ./inputs/train/hr
    dataroot_lq: ./inputs/train/lr
    io_backend:
      type: disk

    gt_size: 512
    use_hflip: true
    use_rot: true

    # data loader
    use_shuffle: true
    num_worker_per_gpu: 4
    batch_size_per_gpu: 4
    dataset_enlarge_ratio: 1
    prefetch_mode: ~

# network structures
network_g:
  type: RRDBNet
  num_in_ch: 3
  num_out_ch: 3
  num_feat: 96
  num_block: 23
  num_grow_ch: 32

network_d:
  type: UNetDiscriminatorSN
  num_in_ch: 3
  num_feat: 96
  skip_connection: True

# path
path:
  # use the pre-trained Real-ESRNet model
  #pretrain_network_g: experiments/pretrained_models/RealESRGAN_x4plus.pth
  pretrain_network_g: ./weights/net_g_905000.pth
  param_key_g: params_ema
  strict_load_g: false
  resume_state: ~
  
# training settings
train:
  ema_decay: 0.999
  optim_g:
    type: Adam
    lr: !!float 1e-4
    weight_decay: 0
    betas: [0.9, 0.99]

  scheduler:
    type: MultiStepLR
    milestones: [400000]
    gamma: 0.5

  total_iter: 999999999
  warmup_iter: -1  # no warm up

  # losses
  pixel_opt:
    type: MSELoss
    loss_weight: 1.0
    reduction: mean

# validation settings
val:
  val_freq: !!float 5e3
  save_img: false
  pbar: False

  metrics:
    psnr:
      type: calculate_psnr
      crop_border: 4
      test_y_channel: true
      better: higher  # the higher, the better. Default: higher
    ssim:
      type: calculate_ssim
      crop_border: 4
      test_y_channel: true
      better: higher  # the higher, the better. Default: higher

# logging settings
logger:
  print_freq: 100
  save_checkpoint_freq: !!float 5e3
  use_tb_logger: true
  wandb:
    project: ~
    resume_id: ~

# dist training settings
dist_params:
  backend: nccl
  port: 29500

```


<br>

## **Installation**

---
1. Clone repo

    ```bash
    git clone https://github.com/Jinwoo1126/Dacon_SR.git
    cd Dacon_SR
    ```
    

2. Install dependent packages

    ```bash
    # Install basicsr - https://github.com/xinntao/BasicSR
    # We use BasicSR for both training and inference
    pip install basicsr
    # facexlib and gfpgan are for face enhancement
    pip install facexlib
    pip install gfpgan
    pip install -r requirements.txt
    python setup.py develop
    ```

<br>

## **Running Code**
---

**Pretrained Model**

[RRDBNet_x4_Pretrained_Model](https://drive.google.com/file/d/1piw_MOIE5bTH3-o9rmWqp3uIZoYcc5Wl/view?usp=sharing)

<br>

**File Structure**


```
./Dacon_SR
├── inputs
│   ├── train
│   │   ├── 0000.png
│   │   ├── 0001.png
│   │   ├── 0002.png
│   │   ├── ...
│   ├── test
│   │   ├── 20000.png
│   │   ├── 20001.png
│   │   ├── ...
├── tags
├── ...
├── weight
│   ├── net_g_905000.pth
```

**Inference**

```bash
python inference_rrdbnetrot.py --model_path=./weights/net_g_905000.pth --input=./inputs/test/lr --suffix=''
```

Results are in the `results` folder


<br>

**Finetuning with pretrained model**

```bash
python realesrgan/train.py -opt options/finetune_realesrnet_x4plus_pairdata.yml
```

<br>



The test result can be **slightly** diff from ours due to the different hardware architectures. 

**reproducible issues**

[https://discuss.pytorch.org/t/different-training-results-on-different-machines-with-simplified-test-code/59378/11](https://discuss.pytorch.org/t/different-training-results-on-different-machines-with-simplified-test-code/59378/11)

[https://discuss.pytorch.org/t/different-result-on-different-gpu/102502](https://discuss.pytorch.org/t/different-result-on-different-gpu/102502)