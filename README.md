### Pytorch differentiable image transform operations
----------------------
All functions are constructed on Pytorch basic tensor operations which supporting autograd backward functions.
Transform functions are tested on CUDA available environment
#### Supporting operations
1. Affine transform
    * Rotation
    * Horizontal flip
    * Vertical flip
    * Translate X
    * Translate Y
    * Sheer X
    * Sheer Y
2. Image enhancement
    * Contrast control
    * Brightness control
-----------------------
#### How to use
~~~python
from transform_module import *

# create new sample image
ori_image = torch.randint((1,3,256,256)).cuda()

# create transform module
# doesn't need to cast transform module to gpu
# all operations conducted with pytorch functional
# takes pytorch tensor
transform_module = Rotation() 

# transform image
trans_image = transform_module(ori_image, param)
~~~