# Assignment 2 Notes

## Assignment Overview 
* Prepare different Deep learning models 
* Generate adversarial samples for those models 
* Make the model robust against those adversaries. 

12 points and 2 bonus points. 

Packages needed : PyTorch (>= 1.13 with GPU support), OpenCV (>=3.0), Numpy, Gdown, Tensorboard. 

Dataset download : Miniplaces dataset.  
```sh
sh download_dataset.sh
```

Write code in student_code.py

Submit 
python3 zip_submission.py

## Dataset

Developed by MIT. Contains 120k images of various scenes in 100 categories. 100k train, 10k validation and 10k test. Top 1/5 accuracy will be validation metric. 

## Tasks

1. CNN
  a. Write convolution (2)
  b. Design and Train CNN 
    i. Train a simple CNN (1)
    ii. Use own convolution (1)
    iii. Design a network (1)
    iv. Use a pre-trained model (1) 

2. ViT
  a. Write attention (2)
  b. Design and implement ViT (2) 

3. Adversarial samples
  a. Generate adversarial samples (2)
  b. Train model against adversarial data (Bonus 2)

## Installation

```
pip3 install torch torchvision torchaudio
pip3 install tensorboard
pip3 install gdown
```

## Code Overview 

Helper code main.py to run the models. In addition, custom_dataloader.py contains dataloader, custom_augment.py contains image augmentation and custom_blocks.py contains transformer blocks.

**main.py** 
* Runs training and validatation. 
* Running on CPU. 
```sh
python3 main.py ../data --epochs=1 --gpu=-1
```

Running SimpleNet
```
python main.py ../data --epochs=60
```

Running with CustomConv

```
python ./main.py ../data --epochs=10 --use-custom-conv
```

Running ViT
```
python ./main.py ../data --epochs=90 --wd 0.05 --lr 0.01 --use-vit
```

Generate adversarial sample 
```
python ./main.py ../data --resume="../logs/torch_conv_exp_2024-10-18_09-41-04/models/model_best.pth.tar" -a -v
```

Training Adv defense model 
```
python main.py ../data --epochs=60 --pgd=True
```

## Adversarial Training
At each batch
  x.requires_grad = True
  pred = model(x)
  loss = loss_fn(pred, target)
  loss.backward()
  (x, dx) -> x
  x.zero_grad()

autograd.grad() allows one to poke on a point in the computational graph to measure its gradient w.r.t another point. 

* Check if x already updated after loss.backward?
* Zero grad x. 

### Adversarial Defense

Q: Does the model learn if num_steps in perturb is 0?
A: Yes it does. 

Sample of input and perturbed input
```
input before perturb tensor([[[[-1.1247, -1.1418, -1.1589,  ..., -0.9192, -0.9363, -0.9705],
          [-1.1760, -1.1760, -1.1760,  ..., -1.0219, -0.9877, -0.9877],
          [-1.2274, -1.2274, -1.2274,  ..., -1.1589, -1.0733, -1.0048],
          ...,
          [-1.0219, -1.0390, -1.0904,  ..., -1.2274, -1.2274, -1.2274],
          [-1.1075, -1.1075, -1.1247,  ..., -1.2617, -1.2788, -1.2788],
          [-1.1760, -1.1589, -1.1418,  ..., -1.2788, -1.2959, -1.2959]],

         [[-1.0203, -1.0378, -1.0553,  ..., -0.8102, -0.8277, -0.8627],
          [-1.0728, -1.0728, -1.0728,  ..., -0.9153, -0.8803, -0.8803],
          [-1.1253, -1.1253, -1.1253,  ..., -1.0553, -0.9678, -0.8978],
          ...,
          [-0.9153, -0.9328, -0.9853,  ..., -1.1253, -1.1253, -1.1253],
          [-1.0028, -1.0028, -1.0203,  ..., -1.1604, -1.1779, -1.1779],
          [-1.0728, -1.0553, -1.0378,  ..., -1.1779, -1.1954, -1.1954]],

         [[-0.7413, -0.7587, -0.7761,  ..., -0.5321, -0.5495, -0.5670],
          [-0.7936, -0.7936, -0.7936,  ..., -0.6367, -0.6018, -0.5844],
          [-0.8458, -0.8458, -0.8458,  ..., -0.7761, -0.6890, -0.6018],
          ...,
          [-0.6193, -0.6541, -0.7064,  ..., -0.8458, -0.8458, -0.8458],
          [-0.7238, -0.7238, -0.7413,  ..., -0.8807, -0.8981, -0.8981],
          [-0.7936, -0.7761, -0.7587,  ..., -0.8981, -0.9156, -0.9156]]]],
       device='cuda:0')
adversary input tensor([[[[-1.1747, -1.0741, -1.2366,  ..., -0.9892, -0.9463, -0.8929],
          [-1.1660, -1.1060, -1.2537,  ..., -0.9919, -1.0177, -0.9377],
          [-1.1774, -1.1497, -1.1574,  ..., -1.2366, -1.1510, -0.9271],
          ...,
          [-0.9442, -0.9614, -1.1404,  ..., -1.1774, -1.1974, -1.2174],
          [-1.1575, -1.1375, -1.1547,  ..., -1.1840, -1.2688, -1.2111],
          [-1.1860, -1.0889, -1.2195,  ..., -1.3088, -1.2182, -1.3659]],

         [[-0.9903, -1.1155, -1.1253,  ..., -0.8002, -0.7500, -0.9404],
          [-1.0828, -1.0228, -0.9951,  ..., -0.9930, -0.8026, -0.8103],
          [-1.0754, -1.0954, -1.2030,  ..., -0.9776, -1.0378, -0.8201],
          ...,
          [-0.9053, -0.9828, -1.0153,  ..., -1.1953, -1.1953, -1.2030],
          [-0.9328, -1.0805, -1.0903,  ..., -1.2304, -1.1479, -1.1002],
          [-0.9951, -1.1253, -1.0878,  ..., -1.2556, -1.2454, -1.2731]],

         [[-0.8113, -0.7087, -0.8261,  ..., -0.5821, -0.6272, -0.5570],
          [-0.8636, -0.7236, -0.8636,  ..., -0.7144, -0.5918, -0.5067],
          [-0.7758, -0.7681, -0.7681,  ..., -0.7461, -0.7390, -0.5241],
          ...,
          [-0.6693, -0.5841, -0.7564,  ..., -0.8558, -0.7958, -0.8158],
          [-0.7938, -0.7938, -0.7913,  ..., -0.8507, -0.9658, -0.8204],
          [-0.7159, -0.6984, -0.7087,  ..., -0.9681, -0.8379, -0.9856]]]],
       device='cuda:0')
```

Q: Does the model learn for very small adversarial changed images? (e.g. diff = ~ 0.01)
A: Seems no.

Q: Does the model learn for 0 adversarial changed images? (diff = 0)
A: 

If no issue, then after 5 epoch in SimpleNet,
******Acc@1 9.310 Acc@5 28.490

With toss < 0.05, with adv training, after 7 epochs,
******Acc@1 14.140 Acc@5 38.960