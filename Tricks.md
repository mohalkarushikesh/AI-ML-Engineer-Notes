##### Tensorflow Installation 
 	- doesn't support 3.13 
	- try: Pip install tensorflow-cpu
	- try: pip install tf-nightly

---

```
import os

# ✅ Set the environment variable
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
print(tf.__version__)

# Run the script 
python -W ignore script.py
```
---
