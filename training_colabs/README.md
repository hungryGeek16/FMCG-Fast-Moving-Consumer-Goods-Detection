# CenterMask:

* Before going into the details of CenterMask's architecture and pipeline, its necessary to understand ```Fully Convolutional One Stage Object Detectors``` and their working principles.

* Mostly, these architectures are used in semantic segmentation tasks, but recently they became popular in object detection tasks due to their simple and anchor-free process. 

## Fully Convolutional One-Stage Object Detection(FCOS):

* This method do not use anchors , which means there would be no RPN layer, which implies no in between IOU thresholding calculations invovled. So, it's basic structure consists 4 modules:

1. BackBone + FPN

2. Classification Head

3. Centerness Head

4. Regression Head

* **These modules can be visualized as given in the diagram below:**

<p align="center">
  <img src="images/1.png" width = 480>
</p>


* The above diagram may give a rough idea about how it works. Here, a fundamental question gets arised, that is ```if there are no anchors , then how would regression head work?``` 

###  Regression Head:

* Regression head produces offsets l(left),r(right),t(top),b(bottom), these are normalized lengths according to pixel at a point.


<p align="center">
  <img src="images/2.png" width = 480>
</p>

* These offsets are applied on to the input image relative to it's size. Hence the output from regression head would be **H X W X 4**.

### Centerness Head:

* Since the process is anchor free, it's cost is impacted while detection. It would produce a dummy box around a true detection.

* The dummy boxes obtained from the regression head would deviate from the object's center which would lower the quality of the detection.

<p align="center">
  <img src="images/3.png" width = 480>
</p>

* Hence a centerness head is added to the overall pipeline, centerness of the ground truth object is calculated and compared with predicted centerness. Then binary cross entropy loss is applied to that difference which is ready to back propagate.

* So while inferencing, centerness scores of particular bounding box is multiplied with its classification score to get a final predicted score. Hence based on these final scores, dummy boxes would be removed in NMS module at the end.

```bash
Centerness formula: sqrt([min(l,r)/max(l,r)] x [min(t,b)/max(t,b)])
```

## Mask-Head in CenterMask:

* When a mask head is added to Faster-RCNN, it becomes Mask RCNN. In the same vein, when mask head is added to FCOS, it becomes CenterMask.

<p align="center">
  <img src="images/4.png" width = 480>
</p>

### Spatial Attention-Guided Mask:

* The intuition behind this module is same as the one metioned [here](https://gitlab.iotiot.in/shunya/products/diamond-detection/issues/10#note_39940).

* After FCOS module, every instance is handled by **ROI-ALIGN** layer, which works adaptively according to the scale of the detected instance. The output from this layer is a feature map of 14 x 14 X C which is fed into 4 consecutive conv layers first.The filter size of these convolutions may vary.

* Then a SAM (spatial attention module) is applied to the output from conv operations. This module consists 4 layers:

1. Parallel Average Pooling and Max Pooling 
 
2. 3 X 3 conv layer

3. Sigmod layer

4. Concatenation of input to SAM with output of SAM.

* After fourth step, a 2 x 2 deconv layer is applied to get the size of 28 x 28 x 80. Finally this goes through 1 x 1 conv layer to get a binary mask.


### Pros and Cons when compared with MASK-RCNN:

#### Pros:

* Training becomes simpler and less time consuming, since RPN and related calculations are not present.

* Higher recall value on detection.

* Attention guided mask head, which makes segmentation efficient.

#### Cons:

* Low precision, which means lower accuracy.

* Detection quality is low when there are overlapping objects.(**Based on observations while training diamond dataset**)

* **Centermask:** [Paper](https://arxiv.org/pdf/1911.06667.pdf)

* **FCOS:** [Paper](https://arxiv.org/pdf/1904.01355.pdf)
