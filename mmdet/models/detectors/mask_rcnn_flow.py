from .two_stage_flow_test import TwoStageDetectorFlowTest
from .two_stage_flow import TwoStageDetectorFlow
from ..registry import DETECTORS


@DETECTORS.register_module
class MaskRCNNFlow(TwoStageDetectorFlowTest):

    def __init__(self,
                 backbone,
                 neck,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 track_head,
                 mask_roi_extractor,
                 mask_head,
                 train_cfg,
                 test_cfg,
                 flow_head=None,
                 pretrained=None):
        super(MaskRCNNFlow, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            track_head=track_head,
            mask_roi_extractor=mask_roi_extractor,
            mask_head=mask_head,
            flow_head=flow_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
