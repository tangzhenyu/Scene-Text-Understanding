=================================== 
Scene Text Recognition based on ctpn and crnn

## ctpn:
- [[github]](https://github.com/tianzhi0549/CTPN)
- Z. Tian, W. Huang, T. He, P. He and Y. Qiao: [Detecting Text in Natural Image with
Connectionist Text Proposal Network, ECCV, 2016.](https://arxiv.org/abs/1609.03605)
## crnn:
- [[github]](https://github.com/bgshih/crnn)
- [An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition](http://arxiv.org/abs/1507.05717)

##Required:
- pytorch
- caffe dependency

## download pretrained model:
- download model from 链接: https://pan.baidu.com/s/1bp8PJBL 密码: qvsj , then copy ctpn_trained_model.caffemodel to ./CTPN.models
- download model from 链接: https://pan.baidu.com/s/1pLmgAvx 密码: a5pq , then copy crnn_trained_model.pth to ./crnn/samples
   
# You can run demo:

  python demo.py
  
  
# Examples:
