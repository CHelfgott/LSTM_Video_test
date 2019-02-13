# -*- coding: utf-8 -*-
"""
Created on Tue May 22 14:07:01 2018
Simple video prediction task -- NN test
@author: Craig
"""

import cv2, numpy as np, os, random, math, glob
import sys
import time
import datetime
import skvideo
skvideo.setFFmpegPath('c:/programdata/anaconda3/library/bin')
import skvideo.io as vidio
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.autograd import Variable
import argparse

NUM_DIMS = 4

INPUT_SIZE = 512
NUM_FRAMES = 1000
NUM_EPOCHS = 100
KERNEL_SIZE = 3
PADDING = KERNEL_SIZE // 2

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


class BouncingBall:
  """ A class to store data on a bouncing ball.

    :param int radius: radius of ball
    :param ndarray([3], int) color: RGB color of ball
    :param ndarray([NUM_DIMS], float) position: position of ball
    :param ndarray([NUM_DIMS], float) velocity: velocity of ball

  """  

  def __init__(self, radius, color, position, velocity):
    self.radius = radius   # Write once
    self.color = color     # Write once
    self.position = position
    self.velocity = velocity
    
  def setPosition(self, new_position):
    self.position = new_position
    
  def setVelocity(self, new_velocity):
    self.velocity = new_velocity
    
  def updatePosition(self):
    self.position += self.velocity
    
  def getPosition(self):
    return self.position
  
  def getVelocity(self):
    return self.velocity
  
  def getRadius(self):
    return self.radius
  
  def getColor(self):
    return self.color
  
  
class BoundingBox:
  """ A class to store data on a bounding box with reflective walls.
      One corner is assumed to be at (0,0).
      Contains utility methods to reflect positions into the box.
  
    :param ndarray([NUM_DIMS], int) corner: position of opposite corner.
    
  """
  def __init__(self, corner):
    self.corner = corner
    
  def isInBounds(self, position, dim):
    return (position[dim] >= 0 and position[dim] <= self.corner[dim])
  
  def reflectIntoBox(self, position):
    new_position = position
    flips = []
    for dim in range(len(self.corner)):
      flips.append(False)
      if not self.isInBounds(position, dim):
        new_position[dim] = position[dim] % (2 * self.corner[dim])
        if new_position[dim] > self.corner[dim]:
          new_position[dim] = 2 * self.corner[dim] - new_position[dim]
          flips[dim] = True

    return new_position, flips
  

def bounceBallInBox(ball, box):
  ball.updatePosition()
  new_position, flips = box.reflectIntoBox(ball.getPosition())
  ball.setPosition(new_position)
  v = ball.getVelocity()
  for dim in range(NUM_DIMS):
    if flips[dim]: v[dim] = -v[dim]
  ball.setVelocity(v)
  

def buildBouncingBallVideo(nBalls, vidSize, nFrames):
  minBallSize = 50
  maxBallSize = int(min(vidSize[0], vidSize[1]) / 3)
  balls = []
  vSizes = [vidSize[0], vidSize[1]]
  for dim in range(2, NUM_DIMS):
    vSizes.append(random.randint(vidSize[0], vidSize[1]))
  box = BoundingBox(vSizes)
  for i in range(nBalls):
    color = [random.randint(0,255),random.randint(0,255),random.randint(0,255)]
    position = np.zeros(NUM_DIMS)
    velocity = np.zeros(NUM_DIMS)
    for dim in range(NUM_DIMS):
      position[dim] = random.uniform(0., vSizes[dim])
      velocity[dim] = random.uniform(-12., 12.)
    balls.append(BouncingBall(random.randint(minBallSize, maxBallSize),
                              color, position, velocity))
  
  videoArray = np.zeros([nFrames, vidSize[1], vidSize[0], 3], dtype=np.uint8)
  for frameCnt in range(nFrames):
    frame = np.zeros([vidSize[1], vidSize[0], 3], dtype=np.uint8)
    for ball in balls:
      position = ball.getPosition()
      relPosition = position[2:] - np.array(vSizes[2:])/2
      planeDistSq = np.sum(relPosition * relPosition)
      center = (int(round(position[0])), int(round(position[1])))
      color = tuple(ball.getColor())
      r = ball.getRadius()
      if r*r > planeDistSq:
        cv2.circle(frame, center, int(round(np.sqrt(r*r - planeDistSq))), color, -1)
      bounceBallInBox(ball, box)     
    videoArray[frameCnt, ...] = frame
  return videoArray


class Conv2DRNN(nn.Module):
  """
  Generate a convolutional RNN cell.
  """
  
  # input size is [batch_size, in_features, height, width]
  # The hidden layer is [batch_size, hidden_layers, height, width]
  # The output layer is [batch_size, out_features, height, width]
  def __init__(self, in_features, hidden_layers, out_features):
    super().__init__()
    self.inf = in_features
    self.hidl = hidden_layers
    self.outf = out_features
    self.Gates = nn.Conv2d(in_features + hidden_layers,
                           hidden_layers + out_features,
                           KERNEL_SIZE, padding=PADDING)
    
  def step(self, input_, prev_hidden):
    # get batch and spatial sizes
    batch_size = input_.data.size()[0]
    spatial_size = input_.data.size()[2:]
    
    # generate empty prev_state, if None is provided
    if prev_hidden is None:
      state_size = [batch_size, self.hidl] + list(spatial_size)
      prev_hidden = Variable(torch.zeros(state_size))
    
    # data size is [batch, channel, height, width]
    stacked_inputs = torch.cat((input_, prev_hidden), 1)
    gates = self.Gates(stacked_inputs)
    activation = nn.LeakyReLu(0.1)
    gates = activation(gates)
    
    hidden = gates[:, 0:self.hidl, :,:]
    output_layer = gates[:, self.hidl:, :,:]
    
    return hidden, output_layer
	
  # Inputs here is [batch_size, num_inputs, input_features, height, width]	
  def forward(self, inputs, hidden=None):
    steps = inputs.data.size()[1]
    outputs = Variable(torch.zeros(list(inputs.data.size()[:2]) + [self.outf] + 
                                   list(inputs.data.size()[3:])))
    for i in range(steps):
      input = inputs[:,i,...]
      hidden, output = self.step(input, hidden)
      outputs[:,i,...] = output.unsqueeze(1)

    return hidden, outputs
    

class VideoNet(nn.Module):
  """
  Generate a network for predicting video
  """
  
  def __init__(self, **kwargs):
    super(VideoNet, self).__init__()
    
    self.rnn_layers = {}
    self.convT = {}
    self.conv = {}
    for i in [3, 6, 12, 24, 48]:
      self.rnn_layers[i] = Conv2DRNN(i, 3, 2*i)
      self.convT[i] = nn.ConvTranspose2D(2*i, i, kernel_size=3, stride=2)
      self.conv[i] = nn.Conv2D(3*i, i, kernel_size=KERNEL_SIZE, padding=PADDING)
      
    self.maxpool = nn.MaxPool2D((2,2))
    self.dropout = nn.DropOut2D(0.1)

  def forward(self, batch_input):
    rnn_outputs = {}
    layer = batch_input  # layer is [NBatch, NFrames, 3, H, W], H = W = size
    height = batch_input.data.size()[3]
    assert(batch_input.data.size()[4] == height)  # height = width
    assert(2 ** math.log(height, 2) == height)    # height is a power of 2
    
    for i in [3, 6, 12, 24]:
      _, rnn_outputs[i] = self.rnn_layers[i].forward(layer)
	  # rnn_outputs is [NBatch, NFrames, 2*i, H(layer), W(layer)]
      layer = torch.stack([self.dropout(self.maxpool(x)) for x in torch.unbind(rnn_outputs[i], 0)], 0)
	  # layer is [NBatch, NFrames, 2*i, H/2, W/2]
    
    for i in [24, 12, 6, 3]:
      layer = torch.stack([self.dropout(self.convT[i].forward(x)) for x in torch.unbind(layer, 0)], 0)
      # layer is [NBatch, NFrames, i, 2*H(layer), 2*W(layer)]
      layer = torch.cat(rnn_outputs[i], layer, 1)
      # layer is [NBatch, NFrames, 3*i, 2*H(layer), 2*W(layer)]
      layer = torch.stack([self.conv[i].forward(x) for x in torch.unbind(layer, 0)], 0)
      # layer is [NBatch, NFrames, i, 2*H(layer), 2*W(layer)]
	
    if not self.training:
      return layer
    
    loss = nn.MSELoss(reduce=False)
    return loss(batch_inputs, layer)
	
  
def train(epoch, video_size, model, optimizer_model, use_gpu,
          samples_per_epoch = 1000, batch_size=25):
  losses = AverageMeter()
  batch_time = AverageMeter()
  data_time = AverageMeter()

  n_iters = samples_per_epoch / batch_size
  for iter in range(n_iters):
    video_inputs = np.zeros([batch_size, NUM_FRAMES, 3, video_size, video_size])
    for i in range(batch_size):
      num_balls = random.randint(4,12)
      video_input = buildBouncingBallVideo(num_balls, 
                                           [video_size, video_size], 
                                           NUM_FRAMES)
      video_inputs[i,...] = video_input
      
    inputs = Variable(torch.from_numpy(video_inputs).float())

    loss = model(inputs)

    optimizer_model.zero_grad()
    loss.backward()
    optimizer_model.step()

    losses[epoch] += loss.data[0]

  if epoch > 0:
    print(epoch, loss.data[0])

  end = time.time()
 
  
def main():
  parser = argparse.ArgumentParser(description='Train video prediction model')
  parser.add_argument('--just_video', type=str2bool, default=False,
                      help="flag to just generate one video")
  parser.add_argument('--size', type=int, default=512,
                      help="size of the square video patch (default: 512)")
  parser.add_argument('--max-epoch', default=NUM_EPOCHS, type=int,
                      help="maximum epochs to run")
  parser.add_argument('--start-epoch', default=0, type=int,
                      help="manual epoch number (useful on restarts)")
  parser.add_argument('--train-batch', default=32, type=int,
                      help="train batch size")
  parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                      help="initial learning rate")                    
  parser.add_argument('--resume', type=str, default='', metavar='PATH')
  parser.add_argument('--evaluate', action='store_true', help="evaluation only")
  parser.add_argument('--save-dir', type=str, default='log')
  parser.add_argument('--use-cpu', action='store_true', help="use cpu")
  parser.add_argument('--gpu-devices', default='0', type=str,
                      help='gpu device ids for CUDA_VISIBLE_DEVICES')
  args = parser.parse_args()
  
  if args.just_video:
    out_video = buildBouncingBallVideo(50, [args.size, args.size], NUM_FRAMES)
    vidio.vwrite("test_video.mp4", out_video)
    return
  
  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
  use_gpu = torch.cuda.is_available()
  if args.use_cpu: use_gpu = False

  if not args.evaluate:
    sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
  else:
    sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
  print("==========\nArgs:{}\n==========".format(args))

  if use_gpu:
    print("Currently using GPU {}".format(args.gpu_devices))
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(args.seed)
  else:
    print("Currently using CPU (GPU is highly recommended)")
	
  print("Initializing model")
  model = VideoNet()
  print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))	
  
  start_epoch = args.start_epoch

  optimizer_model = torch.optim.Adam(model.parameters(), lr=args.lr)
  
  if args.resume:
    print("Loading checkpoint from '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    start_epoch = checkpoint['epoch']

  if use_gpu:
    model = nn.DataParallel(model).cuda()

  if args.evaluate:
    print("Evaluate only")
    test(model, use_gpu)
    return

  start_time = time.time()
  train_time = 0
  best_rank1 = -np.inf
  best_epoch = 0
  print("==> Start training")

  for epoch in range(start_epoch, args.max_epoch):
    start_train_time = time.time()
    train(epoch, arg.size, model, optimizer_model, use_gpu)
    train_time += round(time.time() - start_train_time)
        
    if (epoch+1) % 5 == 0 or (epoch+1) == args.max_epoch:
      print("==> Test")
      rank1 = test(model, args.size, use_gpu)
      is_best = rank1 > best_rank1
      if is_best:
        best_rank1 = rank1
        best_epoch = epoch + 1

      if use_gpu:
        state_dict = model.module.state_dict()
      else:
        state_dict = model.state_dict()
      save_checkpoint({ 'state_dict': state_dict, 'rank1': rank1, 'epoch': epoch,
                      }, is_best, 
                      osp.join(args.save_dir, 'checkpoint_ep' + str(epoch+1) + '.pth.tar'))

    print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    train_time = str(datetime.timedelta(seconds=train_time))
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, train_time))
    
    
if __name__ == '__main__':
  main()