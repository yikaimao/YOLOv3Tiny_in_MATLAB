# YOLOv3-tiny-in-MATLAB

usage: yolov3Tiny_Mao_v2.mlx (OLD)

usage: yolov3Tiny_Mao_v3.mlx (new with batch norm folding)

  1. use out = 1; fold = 0; to output all the learnable parameters into .mat files
  
  2. use BN_fold_single.m to fold the batch norm parameters into conv weights and biases (weights_folded.mat/bias_folded.mat)
  
  3. change to out = 0; fold = 1;

## Required MATLAB Add-ons:

[Computer Vision Toolbox](https://www.mathworks.com/products/computer-vision.html?s_tid=FX_PR_info)

[Deep Learning Toolbox](https://www.mathworks.com/products/deep-learning.html?s_tid=FX_PR_info)

[Image Processing Toolbox](https://www.mathworks.com/products/image.html?s_tid=FX_PR_info)

If you don't have the toolboxes:

[kc_YOLOv3-Tiny](https://github.com/yikaimao/kc_YOLOv3Tiny)

(the final NMS step still uses the Computer Vision Toolbox, but you should have all the raw outputs generated in your workspace.)

## References:

[YOLO: Real-Time Object Detection](https://pjreddie.com/darknet/yolo/)

[Deep Learning : Darknet Importer](https://www.mathworks.com/matlabcentral/fileexchange/71277-deep-learning-darknet-importer)

[Object Detection Using YOLO v3 Deep Learning](https://www.mathworks.com/help/vision/ug/object-detection-using-yolo-v3-deep-learning.html)

[yolov3-yolov4-matlab](https://www.mathworks.com/matlabcentral/fileexchange/75305-yolov3-yolov4-matlab)
