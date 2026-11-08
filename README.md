# Dacon AISR

**4× image super-resolution** for the AI Yangjae Hub Open Source Competition: reconstructing 2048 × 2048 images from 512 × 512 inputs.

[Demo](https://colab.research.google.com/drive/1YsQYNFk4HY_INYjtruLWCcbiZFPJgCP6?usp=sharing) · [Project report](Dacon_AISR_final6.pdf) · [Setup and inference](docs/competition-setup.md) · [Competition](https://dacon.io/competitions/official/235977/overview/description)

| Recorded result | PSNR score | Rank |
| --- | ---: | ---: |
| Private leaderboard | **25.00327** | **1st** |
| Public leaderboard | 25.96657 | 2nd |

| Low-resolution input | 4× reconstruction |
| --- | --- |
| <img src="docs/images/super-resolution-input.png" alt="Two low-resolution street photographs from the demo notebook" width="420"> | <img src="docs/images/super-resolution-output.png" alt="The two corresponding reconstructed street photographs" width="420"> |

*Saved 4× outputs from the [demo notebook](Dacon_AISR_Demo_Description.ipynb). Scores and images are the repository's recorded competition results.*

## Approach

- **RRDBNet reconstruction:** a 23-block network with 64 feature channels and 4× upsampling.
- **Paired training:** low/high-resolution image pairs, rotation and flip augmentation, and 512-pixel ground-truth crops.
- **PSNR-oriented optimization:** MSE loss, Adam, exponential moving-average weights, and a multistep learning-rate schedule.
- **Geometric self-ensemble:** eight rotated/flipped views are restored, transformed back, and averaged in `RRDBNetRot`.

The [training configuration](options/finetune_realesrnet_x4plus_pairdata.yml) and [inference architecture](realesrgan/archs/rrdbnetrot_arch.py) contain the implementation details.

## Try the model

Use the [Colab demo](https://colab.research.google.com/drive/1YsQYNFk4HY_INYjtruLWCcbiZFPJgCP6?usp=sharing), or follow the [local setup guide](docs/competition-setup.md) to install the original dependencies and obtain the competition checkpoint. With `weights/net_g_905000.pth` in place, run:

```bash
python inference_rrdbnetrot.py \
    --model_path=./weights/net_g_905000.pth \
    --input=./inputs/test/lr \
    --output=./results \
    --suffix=''
```

The guide also includes input/output previews, the dataset layout, and fine-tuning instructions.

## Team and acknowledgments

**모두연 포레버** · **박수철 (Soochul Park)**, 장진우, 윤성국, 양성모.

Built on [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) and [BasicSR](https://github.com/XPixelGroup/BasicSR). The [upstream README](resrgan_readme.md) and [BSD 3-Clause license](LICENSE), including the original copyright attribution, are retained.

<details>
<summary>Full project notes</summary>

<!-- preserved-project-notes:start -->
# **Dacon AISR**

[![Colab](https://img.shields.io/static/v1?label=Demo&message=Colab&color=orange)](https://colab.research.google.com/drive/1YsQYNFk4HY_INYjtruLWCcbiZFPJgCP6?usp=sharing)
[![PDF](https://img.shields.io/static/v1?label=View&message=PDF&color=skyblue)](./Dacon_AISR_final6.pdf)
[![Description](https://img.shields.io/static/v1?label=Description&message=Code&color=yellow)](./Dacon_AISR_Demo_Description.ipynb)

**🔥 Team : 모두연 포레버** <br>
**🔥 Member : 박수철, 장진우, 윤성국, 양성모**

<br>

## **Descriptions**
---
**Competition Link : [AI 양재 허브 인공지능 오픈소스 경진대회](https://dacon.io/competitions/official/235977/overview/description)<p>**

**[주제]**

이미지 초해상화(Image Sper-Resolution)를 위한 AI 알고리즘 개발

**[목적 및 배경]**

인공지능 오픈소스를 활용하여 이미지 초해상화 문제를​ 해결함으로써 오픈소스 생태계와 컴퓨터 비전 분야에 기여


**[설명]​**

품질이 저하된 저해상도 촬영 이미지(512x512)를 고품질의 고해상도 촬영 이미지(2048x2048)로 생성

**[평가 산식]​**

PSNR(Peak Signal-to-Noise Ratio)
$$PSNR = 10log_{10}(\frac{R^2}{MSE})$$

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
OS : Ubuntu 20.04.5 LTS
CPU : Intel(R) Xeon(R) Gold 6230R CPU @ 2.10GHz
GPU : A5000 * 4

&

<Sub>
Colab Pro+
```


<br>

## **✅ Competition strategies**
---
**1. Data Augmentation**

 - Rotation

 - Flip

**2. Post Processing**

 - Geometric Self-Ensemble [[https://arxiv.org/pdf/1707.02921.pdf](https://arxiv.org/pdf/1707.02921.pdf)]

**3. Loss function**

 - MSELoss ( PSNR socre is directly related with `MSE` )
 
**4. gt_size increase `gt_size = 64 -> 512`**

**5. Multistep LR optimizer**
 - Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones of 200,000
 - In the second half of the learning, detailed learning is performed.

<br>

**Main configuration & Hyper parameters**
```
manuel_seed : 0
gt_size : 512
num_feat : 64

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
  num_feat: 64
  num_block: 23
  num_grow_ch: 32

network_d:
  type: UNetDiscriminatorSN
  num_in_ch: 3
  num_feat: 64
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
    milestones: [200000]
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

## **🔨 Installation**

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

## **🏃 Running Code**
---

**Pretrained Model**

[RRDBNet_x4_Pretrained_Model](https://drive.google.com/file/d/1piw_MOIE5bTH3-o9rmWqp3uIZoYcc5Wl/view?usp=sharing)

<br>

**File Structure**


```
./Dacon_SR
├── inputs
│   ├── train
|   |   ├── hr
│   │   |   ├── 0000.png
│   │   |   ├── 0001.png
│   │   |   ├── 0002.png
|   |   |    ...
│   │   ├── lr
|   |       ├── 0000.png
|   |       ├── 0001.png
|   |       ├── 0002.png
|   |        ...
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

**(Optional) Finetuning with pretrained model**

```bash
python realesrgan/train.py -opt options/finetune_realesrnet_x4plus_pairdata.yml
```

**experiments result** :  `./experiments/[your experiments name]`

(default='finetune_RealESRNetx4plus_400k_pairdata')

<br>

---
<br>


The test result can be **slightly** diff from ours due to the different hardware architectures.

**reproducible issues**

[https://discuss.pytorch.org/t/different-training-results-on-different-machines-with-simplified-test-code/59378/11](https://discuss.pytorch.org/t/different-training-results-on-different-machines-with-simplified-test-code/59378/11)

[https://discuss.pytorch.org/t/different-result-on-different-gpu/102502](https://discuss.pytorch.org/t/different-result-on-different-gpu/102502)

<!-- preserved-project-notes:end -->

</details>
