# Udacity_AWS_Bike_Sharing
This repo consists of a notebook that utilizes the autogluon framework to train models on historical data. 


To run this notebook ensure to run the imports cell first:

```
!pip install -U pip
!pip install -U setuptools wheel
!pip install -U "mxnet<2.0.0" bokeh>=2.4.3
!pip install autogluon --no-cache-dir
!pip install kaggle
# Without --no-cache-dir, smaller aws instances may have trouble installing
```

If you run into any conflicting libraries, ensure to create a virtual environment and select it as the kernel:
```
python -m venv venv
```

Ensure to also load your own kaggle credentials:
```
# Fill in your user name and key from creating the kaggle account and API token file
import json
kaggle_username = ""
kaggle_key = ""

# Save API token the kaggle.json file
with open("/root/.kaggle/kaggle.json", "w") as f:
    f.write(json.dumps({"username": kaggle_username, "key": kaggle_key}))
```


With your environment set up, feel free to continue runninng the cells in the notebook.😊
