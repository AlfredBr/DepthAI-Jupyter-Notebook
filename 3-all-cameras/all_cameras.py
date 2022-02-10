from pathlib import Path
import blobconverter
import cv2
import depthai
import numpy as np

pipeline = depthai.Pipeline()

cam_rgb = pipeline.createColorCamera()
assert cam_rgb is not None
cam_rgb.setPreviewSize(640, 480)
cam_rgb.setInterleaved(False)

cam_left = pipeline.create(depthai.node.MonoCamera)
assert cam_left is not None
cam_left.setBoardSocket(depthai.CameraBoardSocket.LEFT)
cam_left.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_480_P)

cam_right = pipeline.create(depthai.node.MonoCamera)
assert cam_right is not None
cam_right.setBoardSocket(depthai.CameraBoardSocket.RIGHT)
cam_right.setResolution(depthai.MonoCameraProperties.SensorResolution.THE_480_P)

xout_rgb = pipeline.createXLinkOut()
assert xout_rgb is not None
xout_rgb.setStreamName("rgb")

xout_left = pipeline.createXLinkOut()
assert xout_left is not None
xout_left.setStreamName("left")

xout_right = pipeline.createXLinkOut()
assert xout_right is not None
xout_right.setStreamName("right")

cam_rgb.preview.link(xout_rgb.input)
cam_left.out.link(xout_left.input)
cam_right.out.link(xout_right.input)

with depthai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue("rgb")
    q_left = device.getOutputQueue("left")
    q_right = device.getOutputQueue("right")

    frame_rgb = None
    frame_left = None
    frame_right = None

    fw = 320
    fh = 240

    while True:
        in_rgb = q_rgb.tryGet()
        if in_rgb is not None:
            frame_rgb = in_rgb.getCvFrame()
            frame_rgb = cv2.resize(frame_rgb, (fw, fh))

        in_left = q_left.tryGet()
        if in_left is not None:
            frame_left = in_left.getCvFrame()
            frame_left = cv2.resize(frame_left, (fw, fh))
            frame_left = cv2.cvtColor(frame_left, cv2.COLOR_GRAY2BGR)

        in_right = q_right.tryGet()
        if in_right is not None:
            frame_right = in_right.getCvFrame()
            frame_right = cv2.resize(frame_right, (fw, fh))
            frame_right = cv2.cvtColor(frame_right, cv2.COLOR_GRAY2BGR)

         # concatenate the frames
        if frame_rgb is not None and frame_left is not None and frame_right is not None:
            frame = np.concatenate((frame_left, frame_rgb, frame_right), axis=1)
            cv2.imshow('frame', frame)

        if cv2.waitKey(1) == ord('q'):
            break
