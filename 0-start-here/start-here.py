# first, import all necessary modules
from pathlib import Path
import blobconverter
import cv2
import depthai
import numpy as np

# the pipeline tells DepthAI what operations to perform when running - you define all of the resources used and flows here
pipeline = depthai.Pipeline()

# First, we want the Color camera as the output
colorCamera = pipeline.createColorCamera()
colorCamera.setPreviewSize(300, 300)  # 300x300 will be the preview frame size, available as 'preview' output of the node
colorCamera.setInterleaved(False)

# XLinkOut is a "way out" from the device. Any data you want to transfer to host need to be send via XLink
xout_rgb = pipeline.createXLinkOut()
# For the rgb camera output, we want the XLink stream to be named "rgb"
xout_rgb.setStreamName("rgb")
# Linking camera preview to XLink input, so that the frames will be sent to host
colorCamera.preview.link(xout_rgb.input)

# Pipeline is now finished, and we need to find an available device to run our pipeline
# we are using context manager here that will dispose the device after we stop using it

with depthai.Device(pipeline) as device:
    # From this point, the Device will be in "running" mode and will start sending data via XLink
    # To consume the device results, we get two output queues from the device, with stream names we assigned earlier
    q_rgb = device.getOutputQueue("rgb")
    # Here, some of the default values are defined. Frame will be an image from "rgb" stream, detections will contain nn results
    frame = None
    # Main host-side application loop
    while True:
        # we try to fetch the data from nn/rgb queues. tryGet will return either the data packet or None if there isn't any.
        in_rgb = q_rgb.tryGet()
        if in_rgb is not None:
            # If the packet from RGB camera is present, we're retrieving the frame in OpenCV format using getCvFrame.
            frame = in_rgb.getCvFrame()
        if frame is not None:
            # show the frame on the screen.
            cv2.imshow("preview", frame)
        # at any time, you can press "q" and exit the main loop, therefore exiting the program itself.
        if cv2.waitKey(1) == ord('q'):
            break
