# How to run

Setup
- Create a directory for models
- Copy `models.weight` and `models.cfg` to created directory
- update or create `.env` files to containing model file references
```
MODEL_DIR= "model_dir_name/"
MODEL_WEIGHT="model_weight_name.weights"
MODEL_CONFIG="model_config_name.cfg"
```

Install Requierments
```
pip3 install -r requirements.txt
```

Run API on Local
```
flask run
```

Access on Local
```
uri = http://localhost:5000
```


Request Payload
```
{
    "img": base64encodedImage
}
```

Request Response
```
{
    "img": base64encodedImage
    "label": "label"
}
```
