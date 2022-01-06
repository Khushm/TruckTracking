from loguru import logger
import paddle
import os
from ppdet.slim import build_slim_model
from ppdet.core.workspace import load_config, merge_config
from ppdet.utils.check import check_gpu, check_npu, check_version, check_config
from utils.object_detection.PP_Yolo_Detector.detector_utils.args import parse_args
from utils.object_detection.PP_Yolo_Detector.detector_utils.trainer import Trainer


class PP_Detector:
    def __init__(self, infer_img_path=''):
        try:
            self.infer_img_path = infer_img_path
            self.infer_weights = os.path.join(os.getcwd(), 'models', 'ppyolo_r50vd_dcn_1x_coco.pdparams')
            # self.infer_weights = 'https://paddledet.bj.bcebos.com/models/ppyolo_r50vd_dcn_1x_coco.pdparams'
            self.FLAGS = parse_args(self.infer_img_path)
            self.cfg = load_config(self.FLAGS.config)
            self.cfg['use_vdl'] = self.FLAGS.use_vdl
            self.cfg['vdl_log_dir'] = self.FLAGS.vdl_log_dir
            merge_config(self.FLAGS.opt)

            self.cfg.use_npu = False
            self.cfg.use_gpu = False

            self.place = paddle.set_device('cpu')

            if 'norm_type' in self.cfg and self.cfg['norm_type'] == 'sync_bn' and not self.cfg.use_gpu:
                self.cfg['norm_type'] = 'bn'

            if self.FLAGS.slim_config:
                cfg = build_slim_model(self.cfg, self.FLAGS.slim_config, mode='test')

            check_config(self.cfg)
            check_gpu(self.cfg.use_gpu)
            check_npu(self.cfg.use_npu)
            check_version()

            # # build trainer
            self.trainer = Trainer(self.cfg, mode='test')

            # load weights
            self.trainer.load_weights(self.infer_weights)

            logger.info('Initialised PP_Detection successfully.')

        except Exception as e:
            logger.info(f'Error initialising PP Detector: {e}')

    def pp_infer_image_path(self, image_path, frame):
        try:
            self.frame = frame
            self.sub_dets = []
            self.infer_img_path = image_path
            self.boxes, self.classids, self.confidence = self.trainer.predict([self.infer_img_path])

            if len(self.boxes) > 0:
                for i in range(len(self.classids)):
                    if self.classids[i] != 'truck':
                        continue
                    self.x, self.y, self.w, self.h = self.boxes[i]

                    self.startX = int(self.x)
                    self.endX = int(self.x + self.w)
                    self.startY = int(self.y)
                    self.endY = int(self.y + self.h)

                    self.cords_data = [self.startX, self.startY, self.endX, self.endY, self.confidence[i]]
                    # self.frame = cv2.rectangle(self.frame, (self.startX, self.startY), (self.endX, self.endY), color=(0, 0, 255),
                    #               thickness=1)
                    self.sub_dets.append(self.cords_data)
            return self.sub_dets, self.frame

        except Exception as e:
            logger.debug(f'Error in infernece: {e}')
