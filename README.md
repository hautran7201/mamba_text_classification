
# Mamba text classification

Using the Mamba model to classify the sentiment of IMDb review.


## Installation

Library
```bash
    !pip install datasets evaluate accelerate
    !pip install causal-conv1d>=1.1.0
    !pip install mamba-ssm
    !pip install configargparse
```

Env
```bash
    !echo /usr/lib64-nvidia/ >/etc/ld.so.conf.d/libcuda.conf; ldconfig
```

    
## Running

Train model

```bash
    !python train.py
```

```bash
    !python infer.py --infer_data "I like it"
```



