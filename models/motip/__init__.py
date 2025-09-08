# Copyright (c) Ruopeng Gao. All Rights Reserved.

from .motip import MOTIP
from structures.args import Args
from models.deformable_detr.deformable_detr import build as build_deformable_detr
from models.motip.trajectory_modeling import TrajectoryModeling
from models.motip.id_decoder import IDDecoder

from fast_reid.fastreid.engine import DefaultPredictor
from fast_reid.fastreid.config import get_cfg
import torch.nn as nn

def setup_cfg(config):
    cfg = get_cfg()
    cfg.merge_from_file(config["FAST_REID_CONFIG_PATH"])
    cfg.merge_from_list([])
    cfg.freeze()
    return cfg

def build(config: dict):
    # Generate DETR args:
    detr_args = Args()
    # 1. backbone:
    detr_args.backbone = config["BACKBONE"]
    detr_args.lr_backbone = config["LR"] * config["LR_BACKBONE_SCALE"]
    detr_args.dilation = config["DILATION"]
    # 2. transformer:
    detr_args.num_classes = config["NUM_CLASSES"]
    detr_args.device = config["DEVICE"]
    detr_args.num_queries = config["DETR_NUM_QUERIES"]
    detr_args.num_feature_levels = config["DETR_NUM_FEATURE_LEVELS"]
    detr_args.aux_loss = config["DETR_AUX_LOSS"]
    detr_args.with_box_refine = config["DETR_WITH_BOX_REFINE"]
    detr_args.two_stage = config["DETR_TWO_STAGE"]
    detr_args.hidden_dim = config["DETR_HIDDEN_DIM"]
    detr_args.masks = config["DETR_MASKS"]
    detr_args.position_embedding = config["DETR_POSITION_EMBEDDING"]
    detr_args.nheads = config["DETR_NUM_HEADS"]
    detr_args.enc_layers = config["DETR_ENC_LAYERS"]
    detr_args.dec_layers = config["DETR_DEC_LAYERS"]
    detr_args.dim_feedforward = config["DETR_DIM_FEEDFORWARD"]
    detr_args.dropout = config["DETR_DROPOUT"]
    detr_args.dec_n_points = config["DETR_DEC_N_POINTS"]
    detr_args.enc_n_points = config["DETR_ENC_N_POINTS"]
    detr_args.cls_loss_coef = config["DETR_CLS_LOSS_COEF"]
    detr_args.bbox_loss_coef = config["DETR_BBOX_LOSS_COEF"]
    detr_args.giou_loss_coef = config["DETR_GIOU_LOSS_COEF"]
    detr_args.focal_alpha = config["DETR_FOCAL_ALPHA"]
    detr_args.set_cost_class = config["DETR_SET_COST_CLASS"]
    detr_args.set_cost_bbox = config["DETR_SET_COST_BBOX"]
    detr_args.set_cost_giou = config["DETR_SET_COST_GIOU"]

    detr_framework = config["DETR_FRAMEWORK"].lower()
    match detr_framework:
        case "deformable_detr":
            detr, detr_criterion, _ = build_deformable_detr(args=detr_args)
        case _:
            raise NotImplementedError(f"DETR framework {config['DETR_FRAMEWORK']} is not supported.")

    # Build Fast-ReID predictor:
    cfg, _fastreid_predictor = None, None
    if config["USE_FAST_REID"]:
        cfg = setup_cfg(config)
        _fastreid_predictor = DefaultPredictor(cfg)

    # Build Fast-ReID adaptor:
    class FastReidAdaptor(nn.Module):
        def __init__(self, in_dim = 2048, out_dim=config["FEATURE_DIM"]):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, (in_dim + out_dim) // 2),
                nn.ReLU(),
                nn.Linear((in_dim + out_dim) // 2, (in_dim + out_dim) // 2),
                nn.ReLU(),
                nn.Linear((in_dim + out_dim) // 2, out_dim)
            )
        def forward(self, x):
            return self.net(x)

    _fastreid_adaptor = FastReidAdaptor(in_dim=2048, out_dim=config["FEATURE_DIM"]).to(config["DEVICE"])

    # Build each component:
    # 1. trajectory modeling (currently, only FFNs are used):
    _trajectory_modeling = TrajectoryModeling(
        detr_dim=config["DETR_HIDDEN_DIM"],
        ffn_dim_ratio=config["FFN_DIM_RATIO"],
        feature_dim=config["FEATURE_DIM"],
    ) if config["ONLY_DETR"] is False else None
    # 2. ID decoder:
    _id_decoder = IDDecoder(
        feature_dim=config["FEATURE_DIM"],
        id_dim=config["ID_DIM"],
        ffn_dim_ratio=config["FFN_DIM_RATIO"],
        num_layers=config["NUM_ID_DECODER_LAYERS"],
        head_dim=config["HEAD_DIM"],
        num_id_vocabulary=config["NUM_ID_VOCABULARY"],
        rel_pe_length=config["REL_PE_LENGTH"],
        use_aux_loss=config["USE_AUX_LOSS"],
        use_shared_aux_head=config["USE_SHARED_AUX_HEAD"],
    ) if config["ONLY_DETR"] is False else None

    # Construct MOTIP model:
    motip_model = MOTIP(
        detr=detr,
        detr_framework=detr_framework,
        only_detr=config["ONLY_DETR"],
        fastreid_predictor=_fastreid_predictor,
        fastreid_adaptor=_fastreid_adaptor,
        trajectory_modeling=_trajectory_modeling,
        id_decoder=_id_decoder,
    )

    return motip_model, detr_criterion
