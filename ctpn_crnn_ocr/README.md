=================================== 
Scene Text Recognition based on ctpn and crnn

## ctpn:
- Z. Tian, W. Huang, T. He, P. He and Y. Qiao: [Detecting Text in Natural Image with
Connectionist Text Proposal Network, ECCV, 2016.](https://arxiv.org/abs/1609.03605)
## crnn:
- [An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](http://arxiv.org/abs/1507.05717)

## Data:
- Text detection model are downloaded from [CTPN](https://github.com/tangzhenyu/Scene-Text-Understanding/edit/master/ctpn_crnn_ocr/CTPN)
- Using generated chinese character by using [code](https://github.com/tangzhenyu/Scene-Text-Understanding/tree/master/SynthText_Chinese) to train crnn model.

## Required:
- pytorch
- caffe dependency

## download pretrained model:
- download model from 链接: https://pan.baidu.com/s/1bp8PJBL 密码: qvsj , then copy ctpn_trained_model.caffemodel to ./CTPN.models
- download model from 链接: https://pan.baidu.com/s/1pLmgAvx 密码: a5pq , then copy crnn_trained_model.pth to ./crnn/samples
   
# You can run demo:

  python demo.py

# Tools:
[https://github.com/clcarwin/convert_torch_to_pytorch](https://github.com/clcarwin/convert_torch_to_pytorch)  
  
# Examples:

![Example Image](./01.jpg)
![Example Image](./02.jpg)
