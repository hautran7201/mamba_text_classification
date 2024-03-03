
# Mamba text classification

Using the Mamba model to classify the sentiment of IMDb review.


## Installation

Library
```bash
    !pip install datasets evaluate accelerate
    !pip install causal-conv1d>=1.1.0
    !pip install mamba-ssm
```

Env
```bash
    !export LC_ALL="en_US.UTF-8"
    !export LD_LIBRARY_PATH="/usr/lib64-nvidia"
    !export LIBRARY_PATH="/usr/local/cuda/lib64/stubs"
    !ldconfig /usr/lib64-nvidia
```

    
## Running

Train model

```bash
    !python main.py
```



