#!/usr/bin/python3
from nis import cat
from ssl import HAS_ECDH, _create_unverified_context
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from absl import app
from absl.flags import FLAGS

import rospy
import message_filters
from sensor_msgs.msg import Image
from rospy.numpy_msg import numpy_msg
from geometry_msgs.msg import Vector3
from std_msgs.msg import Header, Int8

# ADD YOLOv5 LIBS #
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

from utils.augmentations import letterbox
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords
from utils.plots import Annotator, colors

coco_id_mapping = {
    1: 'person'
}  # pyformat: disable

coco_ths_mapping = {
     1: 0.7
}  # pyformat: disable

#################################################################################
# Implementation of a CV_Bridge function for compatibility with ROS Melodic
## this is not a complete implementation
## some code lines dealing with exceptions have been removed for convenience

numpy_type_to_cvtype = {'uint8': '8U', 'int8': '8S', 'uint16': '16U',
                        'int16': '16S', 'int32': '32S', 'float32': '32F', 'float64': '64F'}

def cv2_to_imgmsg(cvim, encoding="passthrough", header=None):
    if not isinstance(cvim, (np.ndarray, np.generic)):
        raise TypeError('Your input type is not a numpy array')

    img_msg = Image()
    img_msg.height = cvim.shape[0]
    img_msg.width = cvim.shape[1]

    if header is not None:
        img_msg.header = header

    if cvim.ndim < 3:
        n_channels = 1
    else:
        n_channels = cvim.shape[2]

    cv_type = '%sC%d' % (numpy_type_to_cvtype[cvim.dtype.name], n_channels)

    if encoding == "passthrough":
        img_msg.encoding = cv_type
    else:
        img_msg.encoding = encoding

    if cvim.dtype.byteorder == '>':
        img_msg.is_bigendian = True

    #img_msg.data = cvim.tostring() # Deprecated since numpy version 1.19.0.
    img_msg.data = cvim.tobytes()
    img_msg.step = len(img_msg.data) // img_msg.height

    return img_msg

#################################################################################

class Detector(object):
    def __init__(self):

        # define class variables and set initial values
        ## variables for runnig yolov5 model
        self.image_size = (int(720), int(1280))
        self.conf_thres = 0.15
        self.iou_thres = 0.45
        self.agnostic_nms= False
        self.max_det = 100
        self.max_dist = 3
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        ## variables for topic management
        self.topic_name = {}
        self.topic_name["rgb"] = "/zed2/zed_node/left/image_rect_color"
        self.topic_name["depth"] = "/zed2/zed_node/depth/depth_registered"
        self.raw_img = {}
        self.raw_img["rgb"] = np.zeros( self.image_size + (3,) )
        self.raw_img["depth"] = np.zeros( self.image_size )
        self.is_received = {}
        self.is_received["rgb"] = False
        self.is_received["depth"] = False
        self.rcv_time = {}
        self.rcv_time["rgb"] = None
        self.rcv_time["depth"] = None
        self.proc_time = rospy.get_time()
        
        ## variables for controlling module status
        self.model = None
        self.is_initialized = False
        ## variables for camera to world transformation
        # self.color_intrinsic = np.array([[337.2084410968044, 0.0,               320.5], 
        #                                  [0.0,               337.2084410968044, 240.5], 
        #                                  [0.0,               0.0,               1.0  ]])
        # self.inverse_intrinsic = np.linalg.inv(self.color_intrinsic)
        # self.cam_to_body = np.array([[ 0,  0,  1],
        #                              [-1,  0,  0],
        #                              [ 0, -1,  0]])

        # SUBSCRIBERS
        self.sub_rgb = message_filters.Subscriber(self.topic_name["rgb"], numpy_msg(Image))
        self.sub_depth = message_filters.Subscriber(self.topic_name["depth"], numpy_msg(Image))
        self.sub_ts = message_filters.TimeSynchronizer([self.sub_rgb, self.sub_depth], queue_size=10)
        self.sub_ts.registerCallback(self.sync_callback)
        # PUBLISHERS
        self.pub_img = rospy.Publisher("/detection/image_rect", Image, queue_size=1)
        self.pub_crowd = rospy.Publisher("/detection/num_people", Int8, queue_size=1)

        # initialize the yolov5 model
        self._init_model()

    @torch.no_grad()
    def _init_model(self):
        # initialize model and load pretrained weight
        print('Initializing model...')
        self.model = attempt_load('yolov5m.pt', map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        self.model.eval()
        print('model loaded')
        if self.is_initialized == False:
            self.is_initialized = True

    def _preprocess_data(self, input_img):
        ## padding and resizing
        tmp_img = letterbox(input_img, self.image_size, stride=self.stride, auto=True)[0]
        ## switching axes
        tmp_img = tmp_img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        tmp_img = np.ascontiguousarray(tmp_img)
        ## noramlization
        tmp_img = tmp_img / 255.0
        ## convert data format from numpy to tensor
        input_tensor = torch.from_numpy(tmp_img).unsqueeze(dim=0).to(self.device).float()
        return input_tensor

    def _make_prediction(self, input_tensor):
        # make prediction for the current input
        pred, _ = self.model(input_tensor)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, \
                None, self.agnostic_nms, max_det=self.max_det)
        ## get detection result of the first sample
        ## since only one sample is given as input
        result = pred[0]

        return result

    def _measure_distance(self, box, depth_img):
        H, W = self.image_size
        # find 3d position of the center point of detected box
        box = [int(val) for val in box]
        x_0, y_0, x_1, y_1 = box

        ## slice the center box
        img_slice = depth_img[y_0:y_1, x_0:x_1]
        ## if over 50% of values in the center box is NaN, skip the current box
        nan_idx = np.isnan(img_slice)
        if np.mean(nan_idx) < 0.5:
            distance = np.mean(img_slice[np.invert(nan_idx)])
        else:
            distance = np.nan

        return distance

    def _process_detections(self, annotator, detections, tensor_size, depth_img):
        # initilaize a variable to store number of people in the image
        num_people = 0
        ## scale detection box cooridnates to fit the input image size
        detections[:, :4] = scale_coords(tensor_size, detections[:, :4], self.image_size).round()
        ## draw detection boxes
        for *xyxy, conf, obj in reversed(detections):
            # get the class index
            c = int(obj)
            c_coco = c+1

            # check if the detected class is in the coco_id_mapping
            if not (c_coco in coco_id_mapping.keys()):
                continue
            # get the name for the detected class
            name = coco_id_mapping[c_coco]
            # get the confidence threshold
            conf_ths = coco_ths_mapping[c_coco]
            # check if the confidence score is over the threshold
            if conf.data.cpu() < conf_ths:
                continue
            
            # measure distance values in the box
            distance = self._measure_distance(xyxy, depth_img)
            if distance > self.max_dist:
                continue

            # draw a detection box
            color = colors(c, bgr=True)
            annotator.box_label(xyxy, name, color=color)
            num_people += 1

        return num_people

    def sync_callback(self, rgb, depth):
        try:
            # store the rgb message
            self.raw_img["rgb"] = np.frombuffer(rgb.data, dtype=np.uint8).reshape(self.image_size+(3,))
            self.is_received["rgb"] = True
            self.rcv_time["rgb"] = rgb.header.stamp
            # store the depth message
            self.raw_img["depth"] = np.frombuffer(depth.data, dtype=np.float32).reshape(self.image_size)
            self.is_received["depth"] = True
            self.rcv_time["depth"] = depth.header.stamp

        except Exception as e:
            print(e)
        
    def run_model(self):
        # if no image has been received, skip the loop
        if (self.is_received["rgb"] and self.is_received["depth"]) is False:
            return
        
        # measure the start time of current loop
        t0 = rospy.get_time()

        # if the last process time has not been outdated than a second, skip the loop
        # if (t0 - self.proc_time) < 1.0:
        #     return

        # copy the recent images
        tmp_img = np.copy(self.raw_img["rgb"][:, :, :3])
        tmp_depth = np.copy(self.raw_img["depth"])
        tmp_stamp = self.rcv_time["rgb"]
        ## store the time stamp into a header
        header = Header()
        header.stamp = tmp_stamp

        # prepare tensor input data
        tmp_img_ = self._preprocess_data(tmp_img)

        # make prediction
        result = self._make_prediction(tmp_img_)

        # process the detection result
        ## define a annotator object for drawing detection boxes
        annotator = Annotator(tmp_img, line_width=1, pil=False)
        num_people = self._process_detections(annotator, result, tmp_img_.shape[2:], tmp_depth)
        ## get the annotation result
        output_img = annotator.result()

        # measure the time when the detection is finished
        t1 = rospy.get_time()

        # print("inference time: %4.3f"%(t1-t0))

        # publish the detection result
        self.pub_img.publish(cv2_to_imgmsg(output_img.astype(np.uint8), "rgb8", header))
        self.pub_crowd.publish(Int8(data=num_people))

        # update the last process time
        self.proc_time = t1

def main(_argv):
    rospy.init_node('simple_crowd_detector')

    detector = Detector()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        if detector.is_initialized is True:
            detector.run_model()
            rate.sleep()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit as e:
        print(e)

