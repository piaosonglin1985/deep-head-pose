import os
import time
import csv
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import hopenet, utils
import cv2
import dlib

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


model_path='/home/songlin/deeplearning/deep-head-pose/output/snapshots/mobilenet2_epoch_25.pkl'
face_model_path='/home/songlin/deeplearning/head_pose/mmod_human_face_detector.dat'

def main():

    #can improve speed tremendously
    torch.backends.cudnn.benchmark = True

    assert os.path.isfile(model_path), \
        "=> no model found at '{}'".format(model_path)
    print("=> loading model '{}'".format(model_path))

    # MobilNet structure
    model = hopenet.MobileNet2(66, 3, False)

    print 'Loading snapshot.'
    # Load snapshot
    saved_state_dict = torch.load(model_path)
    model.load_state_dict(saved_state_dict)

    # Dlib face detection model
    cnn_face_detector = dlib.cnn_face_detection_model_v1(face_model_path)

    transformations = transforms.Compose([transforms.Scale(224),
                                          transforms.CenterCrop(224), transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    gpu = 0
    model.cuda(gpu)

    print 'Ready to test network.'

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    idx_tensor = [idx for idx in xrange(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

    cap = cv2.VideoCapture(0)

    while(1):

        ret, frame = cap.read()
        cv2_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Dlib detect
        dets = cnn_face_detector(cv2_frame, 1)

        for idx, det in enumerate(dets):
            # Get x_min, y_min, x_max, y_max, conf
            x_min = det.rect.left()
            y_min = det.rect.top()
            x_max = det.rect.right()
            y_max = det.rect.bottom()
            conf = det.confidence

            if conf > 1.0:
                bbox_width = abs(x_max - x_min)
                bbox_height = abs(y_max - y_min)
                x_min -= 2 * bbox_width / 4
                x_max += 2 * bbox_width / 4
                y_min -= 3 * bbox_height / 4
                y_max += bbox_height / 4
                x_min = max(x_min, 0)
                y_min = max(y_min, 0)
                x_max = min(frame.shape[1], x_max)
                y_max = min(frame.shape[0], y_max)
                # Crop image
                img = cv2_frame[y_min:y_max, x_min:x_max]
                img = Image.fromarray(img)

                # Transform
                img = transformations(img)
                img_shape = img.size()
                img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
                img = Variable(img).cuda(gpu)

                # compute output
                end = time.time()
                with torch.no_grad():
                    yaw, pitch, roll = model(img)
                gpu_time = time.time() - end

                print("gpu time is %f ms.\n" % (gpu_time * 1000))

                yaw_predicted = F.softmax(yaw)
                pitch_predicted = F.softmax(pitch)
                roll_predicted = F.softmax(roll)

                # Get continuous predictions in degrees.
                yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
                pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
                roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99


                #utils.plot_pose_cube(frame, yaw_predicted, pitch_predicted, roll_predicted, (x_min + x_max) / 2, (y_min + y_max) / 2, size = bbox_width)
                utils.draw_axis(frame, yaw_predicted, pitch_predicted, roll_predicted, tdx=(x_min + x_max) / 2,
                                tdy=(y_min + y_max) / 2, size=bbox_height / 2)

                cv2.imshow("result", frame)
                cv2.waitKey(2)
                # Plot expanded bounding box
                # cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)

    out.release()
    video.release()


if __name__ == '__main__':
    main()
