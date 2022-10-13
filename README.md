Team : 

**Public Score**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1d3eca25-7532-4884-80f0-cfeef72e41cb/Untitled.png)

**Private Score**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/70f759da-c296-4b84-bd20-d215473790e4/Untitled.png)

**Environment**

> OS : 
CPU :
GPU : A100 * 8
Memory :
> 

&

**Colab Pro+**

**File Structure**

```
./Real-ESRGAN
├── inputs
│   ├── train
│   │   ├── 0000.png
│   │   ├── 0001.png
│   │   ├── 0002.png
│   │   ├── ...
│   ├── test
│   │   ├── 
│   │   ├──
│   │   ├── ...
├── tags
├── weight
│   ├── 905000.pth
```

Base **Model** 

 - RRDBNet

**Data Augmentation**

 - Rotation

 - Flip

Post Processing

 - Geometric Self-Ensemble [[https://arxiv.org/pdf/1707.02921.pdf](https://arxiv.org/pdf/1707.02921.pdf)]

**Configuration**

|  The test result can be **slightly** diff from ours due to the different hardware architectures. 

**reproducible issue**

[https://discuss.pytorch.org/t/different-training-results-on-different-machines-with-simplified-test-code/59378/11](https://discuss.pytorch.org/t/different-training-results-on-different-machines-with-simplified-test-code/59378/11)

[https://discuss.pytorch.org/t/different-result-on-different-gpu/102502](https://discuss.pytorch.org/t/different-result-on-different-gpu/102502)