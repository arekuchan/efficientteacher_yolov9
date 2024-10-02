import copy
# from .shufflenetv2 import ShuffleNetV2
from .yolov5_backbone import YoloV5BackBone
from .yolov9-tiny_backbone import YoloV9TinyBackBone
# from .cspresnet import CSPResNet
# from .oneshot_shuffle import UltraShuffleNetOneShot
# from .efficientrep import EfficientRep
# from .yolov7_backbone import YoloV7BackBone
# from .resnet import resnet50
def build_backbone(cfg):
    backbone_cfg = copy.deepcopy(cfg)
    # name = backbone_cfg.pop("name")
    name = backbone_cfg.Model.Backbone.name
    if name == "YoloV5":
        return YoloV5BackBone(backbone_cfg)
    elif name == "YoloV9-tiny":
        return YoloV9TinyBackBone(backbone_cfg)
    else:
        raise NotImplementedError