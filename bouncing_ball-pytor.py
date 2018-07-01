# -*- coding: utf-8 -*-
"""
Created on Tue May 22 14:07:01 2018
Simple video prediction task -- NN test
@author: Craig
"""

import cv2, numpy as np, os, random, math, glob
import skvideo.io as vidio
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
from torch.autograd import Variable
import argparse

INPUT_SIZE = 512
NUM_FRAMES = 50
NUM_EPOCHS = 100
KERNEL_SIZE = 3
PADDING = KERNEL_SIZE // 2

class BouncingBall:
  """ A class to store data on a bouncing ball.

    :param int radius: radius of ball
    :param ndarray([3], int) color: RGB color of ball
    :param ndarray([2], float) position: XY position of ball
    :param ndarray([2], float) velocity: XY velocity of ball

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
      Southeast corner is assumed to be at (0,0).
      Contains utility methods to reflect positions into the box.
  
    :param ndarray([2], int) corner: XY position of northeast corner.
    
  """
  def __init__(self, corner):
    self.corner = corner
    
  def isInXBounds(self, position):
    return (position[0] >= 0 and position[0] <= self.corner[0])
  
  def isInYBounds(self, position):
    return (position[1] >= 0 and position[1] <= self.corner[1])
  
  def reflectIntoBox(self, position):
    new_position = position
    x_flip = False
    y_flip = False
    if not self.isInXBounds(position):
      new_position[0] = position[0] % (2 * self.corner[0])
      if new_position[0] > self.corner[0]:
        new_position[0] = 2 * self.corner[0] - new_position[0]
        x_flip = True
    if not self.isInYBounds(position):
      new_position[1] = position[1] % (2 * self.corner[1])
      if new_position[1] > self.corner[1]:
        new_position[1] = 2 * self.corner[1] - new_position[1]
        y_flip = True
    return new_position, x_flip, y_flip
  

def bounceBallInBox(ball, box):
  ball.updatePosition()
  new_position, x_flip, y_flip = box.reflectIntoBox(ball.getPosition())
  ball.setPosition(new_position)
  if x_flip or y_flip:
    v = ball.getVelocity()
    if x_flip: v[0] = -v[0]
    if y_flip: v[1] = -v[1]
    ball.setVelocity(v)
  

def buildBouncingBallVideo(nBalls, vidSize, nFrames):
  box = BoundingBox(vidSize)
  minBallSize = 16
  maxBallSize = min(vidSize[0], vidSize[1]) / 16
  balls = []
  for i in range(nBalls):
    color = [random.randint(0,255),random.randint(0,255),random.randint(0,255)]
    position = np.array([random.uniform(0.,vidSize[0]), 
                         random.uniform(0.,vidSize[1])])
    velocity = np.array([random.uniform(-15.,15.), random.uniform(-15.,15.)])
    balls.append(BouncingBall(random.randint(minBallSize, maxBallSize),
                              color, position, velocity))
  
  videoArray = np.zeros([nFrames, vidSize[1], vidSize[0], 3], dtype=np.uint8)
  for frameCnt in range(nFrames):
    frame = np.zeros([vidSize[1], vidSize[0], 3], dtype=np.uint8)
    for ball in balls:
      position = ball.getPosition()
      position = (int(round(position[0])), int(round(position[1])))
      color = tuple(ball.getColor())
      cv2.circle(frame, position, ball.getRadius(), color, -1)
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
  
  def __init__(self, size, lr=0.0001):
	super(VideoNet, self).__init__()
    assert(2 ** math.log(size, 2) == size) # size is a base 2 power
    self.size = size
    self.lr = lr
    
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
	  
	return layer
	
def train(start_iter, end_iter, size=512, model, batch_size=25, num_per_epoch=1000, epochs=100):
  loss = nn.MSELoss(reduce=False)
  