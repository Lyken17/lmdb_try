import json

with open("config.json", 'r') as fp:
    config = json.load(fp)

caffe_path = config["path"]["caffe"]
data_path = config["path"]["data_folder"]
augment_flip = config["augment"]["flip"]