from utils.object_detection.PP_Yolo_Detector.detector_utils.cli import ArgsParser
from loguru import logger

def parse_args(infer_img_path):
    try:
        parser = ArgsParser()
        parser.add_argument(
            "--infer_dir",
            type=str,
            default=None,
            help="Directory for images to perform inference on.")
        parser.add_argument(
            "--infer_img",
            type=str,
            default=infer_img_path,
            help="Image path, has higher priority over --infer_dir")
        parser.add_argument(
            "--output_dir",
            type=str,
            default="output",
            help="Directory for storing the output visualization files.")
        parser.add_argument(
            "--draw_threshold",
            type=float,
            default=0.5,
            help="Threshold to reserve the result for visualization.")
        parser.add_argument(
            "--slim_config",
            default=None,
            type=str,
            help="Configuration file of slim method.")
        parser.add_argument(
            "--use_vdl",
            type=bool,
            default=False,
            help="Whether to record the data to VisualDL.")
        parser.add_argument(
            '--vdl_log_dir',
            type=str,
            default="vdl_log_dir/image",
            help='VisualDL logging directory for image.')
        parser.add_argument(
            "--save_txt",
            type=bool,
            default=False,
            help="Whether to save inference result in txt.")
        args = parser.parse_args()
        return args
    except Exception as e:
        logger.info(f'Error in Loading FLAGS: {e}')