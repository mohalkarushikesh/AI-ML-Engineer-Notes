##### Tensorflow Installation 
```
	- pip install tensorflow
	- doesn't support 3.13 
	- try: pip install tensorflow-cpu
	- try: pip install tf-nightly

import os

# ✅ Set the environment variable
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
print(tf.__version__)

# Run the script 
python -W ignore script.py
```
---

##### PySpark installation 
```
pip install pyspark 
try: pip install --user pyspark 
try: pip cache purge 
try: pip install --user --no-cache-dir pyspark 
try: pip install --user pyspark
```
