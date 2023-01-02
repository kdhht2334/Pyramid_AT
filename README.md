## PyramidAT (PyTorch version)

<p align="center">
<img src="https://github.com/kdhht2334/Pyramid_AT/blob/main/pics/door.png" height="400", width="3000"/>
</p>

(Original) Paper
---
[Pyramid Adversarial Training Improves ViT's Performance](https://arxiv.org/abs/2111.15121)

Official repository with TF/JAX
---
[link](https://github.com/google-research/scenic/tree/main/scenic/projects/adversarialtraining)


Usage
---

All you have to do is loading images and deep model(s).

```python
import torchvision
import cv2

from pyramidAT import pyramidAT

n_steps = 10

models = torchvision.models.resnet50(pretrained='imagenet').eval()
images = cv2.resize(cv2.imread('imgs/golf_ball.jfif'), (224,224))

perturbed_image = pyramidAT(images, model, mode='nearest', n_steps=n_steps)
```

Milestone
---
  - [x] Convert original TF/JAX to PyTorch version
  - [ ] PyramidAT with dataloader
  - [ ] Bench-marking table
  

Contact (or collaborate)
---
email: `kdhht5022@gmail.com`