import cv2
from tensorflow.python.client import device_lib

from openpose import net

print(cv2.cuda.getCudaEnabledDeviceCount())

print(device_lib.list_local_devices())
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)