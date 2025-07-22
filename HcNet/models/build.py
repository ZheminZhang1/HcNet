
from .HcNet import HcNet,GroupNorm



def build_model(config, is_pretrain=False):

    layernorm = GroupNorm
    model = HcNet(img_size=config.DATA.IMG_SIZE,
                            patch_size=config.MODEL.SWIN.PATCH_SIZE,
                            in_chans=config.MODEL.SWIN.IN_CHANS,
                            num_classes=config.MODEL.NUM_CLASSES,
                            embed_dim=config.MODEL.SWIN.EMBED_DIM,
                            depths=config.MODEL.SWIN.DEPTHS,
                            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                            drop_rate=config.MODEL.DROP_RATE,
                            drop_path_rate=config.MODEL.DROP_PATH_RATE,
                            norm_layer=layernorm,
                            patch_norm=config.MODEL.SWIN.PATCH_NORM,
                            use_checkpoint=config.TRAIN.USE_CHECKPOINT)



    return model
