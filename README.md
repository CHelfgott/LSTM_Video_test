This repo contains some basic experiments in video prediction.

It contains a basic bouncing-balls video generator and an implementation of Convolutional RNN made by [me](https://github.com/CHelfgott).
It also contains an implementation of Convolutional LSTM in PyTorch cloned from 
[here](https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py), 
made by [Andrea Palazzi](https://github.com/ndrplz) and [DavideA](https://github.com/DavideA).

Please note that the ConvLSTM implements the following dynamics:
![CLSTM_dynamics](https://user-images.githubusercontent.com/7113894/59357391-15c73e00-8d2b-11e9-8234-9d51a90be5dc.png)

which is a bit different from the one in the original [paper](https://arxiv.org/pdf/1506.04214.pdf).


### Disclaimer

This is still a work in progress and is far from being perfect: if you find any bug please don't hesitate to open an issue.
