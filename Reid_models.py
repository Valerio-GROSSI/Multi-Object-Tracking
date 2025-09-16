# List of available ReID torchreid models for use with DeepSORT
# from torchreid import models
# models.show_avai_models()
DEEPSORT_REID_MODELS = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 
 'resnext50_32x4d', 'resnext101_32x8d', 'resnet50_fc512', 'se_resnet50', 
 'se_resnet50_fc512', 'se_resnet101', 'se_resnext50_32x4d', 'se_resnext101_32x4d', 
 'densenet121', 'densenet169', 'densenet201', 'densenet161', 'densenet121_fc512', 
 'inceptionresnetv2', 'inceptionv4', 'xception', 'resnet50_ibn_a', 'resnet50_ibn_b', 
 'nasnsetmobile', 'mobilenetv2_x1_0', 'mobilenetv2_x1_4', 'shufflenet', 'squeezenet1_0', 
 'squeezenet1_0_fc512', 'squeezenet1_1', 'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 
 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0', 'mudeep', 'resnet50mid', 'hacnn', 
 'pcb_p6', 'pcb_p4', 'mlfn', 'osnet_x1_0', 'osnet_x0_75', 'osnet_x0_5', 'osnet_x0_25', 
 'osnet_ibn_x1_0', 'osnet_ain_x1_0', 'osnet_ain_x0_75', 'osnet_ain_x0_5', 'osnet_ain_x0_25']

# List of ReID model weights URLs that could be used if specified in CLI for any tracker except byteTrack
# Expand the list with more models as needed if not already present in the Embedding_models folder
REID_MODEL_WEIGHTS_URLS = {
    # osnet
    "osnet_x1_0_market1501.pt": "https://drive.google.com/uc?id=1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA",
    "osnet_x1_0_dukemtmcreid.pt": "https://drive.google.com/uc?id=1QZO_4sNf4hdOKKKzKc-TZU9WW1v6zQbq",
    "osnet_x1_0_msmt17.pt": "https://drive.google.com/uc?id=112EMUfBPYeYg70w-syK6V6Mx8-Qb9Q1M",
    "osnet_x0_75_market1501.pt": "https://drive.google.com/uc?id=1ozRaDSQw_EQ8_93OUmjDbvLXw9TnfPer",
    "osnet_x0_75_dukemtmcreid.pt": "https://drive.google.com/uc?id=1IE3KRaTPp4OUa6PGTFL_d5_KQSJbP0Or",
    "osnet_x0_75_msmt17.pt": "https://drive.google.com/uc?id=1QEGO6WnJ-BmUzVPd3q9NoaO_GsPNlmWc",
    "osnet_x0_5_market1501.pt": "https://drive.google.com/uc?id=1PLB9rgqrUM7blWrg4QlprCuPT7ILYGKT",
    "osnet_x0_5_dukemtmcreid.pt": "https://drive.google.com/uc?id=1KoUVqmiST175hnkALg9XuTi1oYpqcyTu",
    "osnet_x0_5_msmt17.pt": "https://drive.google.com/uc?id=1UT3AxIaDvS2PdxzZmbkLmjtiqq7AIKCv",
    "osnet_x0_25_market1501.pt": "https://drive.google.com/uc?id=1z1UghYvOTtjx7kEoRfmqSMu-z62J6MAj",
    "osnet_x0_25_dukemtmcreid.pt": "https://drive.google.com/uc?id=1eumrtiXT4NOspjyEV4j8cHmlOaaCGk5l",
    "osnet_x0_25_msmt17.pt": "https://drive.google.com/uc?id=1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF",
    # mobilenetv2
    "mobilenetv2_x1_0_market1501.pt": "https://drive.google.com/uc?id=18DgHC2ZJkjekVoqBWszD8_Xiikz-fewp",
    "mobilenetv2_x1_0_dukemtmcreid.pt": "https://drive.google.com/uc?id=1q1WU2FETRJ3BXcpVtfJUuqq4z3psetds",
    "mobilenetv2_x1_0_msmt17.pt": "https://drive.google.com/uc?id=1j50Hv14NOUAg7ZeB3frzfX-WYLi7SrhZ",
    "mobilenetv2_x1_4_market1501.pt": "https://drive.google.com/uc?id=1t6JCqphJG-fwwPVkRLmGGyEBhGOf2GO5",
    "mobilenetv2_x1_4_dukemtmcreid.pt": "https://drive.google.com/uc?id=12uD5FeVqLg9-AFDju2L7SQxjmPb4zpBN",
    "mobilenetv2_x1_4_msmt17.pt": "https://drive.google.com/uc?id=1ZY5P2Zgm-3RbDpbXM0kIBMPvspeNIbXz",
    # lmbn
    "lmbn_n_duke.pt": "https://github.com/mikel-brostrom/yolov8_tracking/releases/download/v9.0/lmbn_n_duke.pth",
    "lmbn_n_market.pt": "https://github.com/mikel-brostrom/yolov8_tracking/releases/download/v9.0/lmbn_n_market.pth",
    "lmbn_n_cuhk03_d.pt": "https://github.com/mikel-brostrom/yolov8_tracking/releases/download/v9.0/lmbn_n_cuhk03_d.pth"
}
