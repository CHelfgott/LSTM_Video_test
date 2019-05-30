# -*- coding: utf-8 -*-
"""
Created on Tue May 22 14:07:01 2018
Simple video prediction task -- NN test
@author: Craig
"""

import cv2, numpy as np, os, random, math
import os.path as osp
import time
import datetime
import skvideo.io as vidio
import torch, torch.nn as nn, torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import argparse

NUM_DIMS = 3

INPUT_SIZE = 128
NUM_FRAMES = 100
NUM_EPOCHS = 100
KERNEL_SIZE = 3
PADDING = KERNEL_SIZE // 2

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self, name, fmt=':.3f'):
    self.name = name
    self.fmt = fmt
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count if self.count != 0 else 0

  def __str__(self):
    fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
    return fmtstr.format(**self.__dict__)    

    
class ProgressMeter(object):
  def __init__(self, num_batches, prefix="", *meters):
    self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
    self.meters = meters
    self.prefix = prefix

  def printb(self, batch):
    entries = [self.prefix + self.batch_fmtstr.format(batch)]
    entries += [str(meter) for meter in self.meters]
    print('\t'.join(entries))

  def _get_batch_fmtstr(self, num_batches):
    num_digits = len(str(num_batches // 1))
    fmt = '{:' + str(num_digits) + 'd}'
    return '[' + fmt + '/' + fmt.format(num_batches) + ']'


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
  
# Assume lightDirection is an ndarray 3-vector with positive z-component.
def drawBall(ball, vSizes, lightDirection, frame):
  position = ball.getPosition()
  color = tuple(ball.getColor())
  r = ball.getRadius()
  center = (int(round(position[0])), int(round(position[1])))

  if len(lightDirection) != 3 or lightDirection[2] <= 0 or len(vSizes) < 3:
    relPosition = position[2:] - np.array(vSizes[2:])/2
    planeDistSq = np.sum(relPosition * relPosition)
    if r*r <= planeDistSq: return
    inPlaneRadius = int(round(np.sqrt(r*r - planeDistSq)))
    cv2.circle(frame, center, inPlaneRadius, color, -1)
    return
    
  relPosition = position[3:] - np.array(vSizes[3:])/2
  distSq = np.sum(relPosition * relPosition)
  if r*r <= distSq: return
  newRadius = int(np.sqrt(r*r - distSq))
  xmin = max(0, center[0] - newRadius)
  xmax = min(vSizes[0], center[0] + newRadius)
  ymin = max(0, center[1] - newRadius)
  ymax = min(vSizes[1], center[1] + newRadius)
  
  xyMesh = np.mgrid[xmin-center[0] : xmax-center[0], ymin-center[1] : ymax-center[1]]
  rSqFn = np.sum(xyMesh * xyMesh, axis=0)
  dzSqFn = np.maximum((r*r - distSq) * np.ones(rSqFn.shape) - rSqFn, np.zeros(rSqFn.shape))
  dzFn = np.sqrt(dzSqFn)
  incidence = xyMesh[0] * lightDirection[0] + xyMesh[1] * lightDirection[1] - dzFn * lightDirection[2]
  incidence /= newRadius * np.sqrt(np.sum(lightDirection * lightDirection))
  toEye = dzFn / newRadius
  fadeFactor = 0.05 * (dzFn > 0)
  fadeFactor -= 0.95 * incidence * (incidence < 0) * (dzFn > 0)
  fadeFactor *= toEye

  newColor = np.stack([color[i] * fadeFactor for i in range(3)], axis=2)
  frame[xmin:xmax, ymin:ymax, :] = np.where(
    np.expand_dims(dzFn, axis=2) > 0, newColor, frame[xmin:xmax, ymin:ymax, :])

  
def buildBouncingBallVideo(nBalls, vidSize, nFrames):
  print("Building video with {} balls, size {}, {} frames".format(nBalls, vidSize, nFrames))
  minBallSize = 25
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
      velocity[dim] = random.uniform(-vSizes[dim]/(np.sqrt(NUM_DIMS) * 32.),
                                     vSizes[dim]/(np.sqrt(NUM_DIMS) * 32.))
    velocity[0] = random.uniform(-vSizes[0]/32., vSizes[0]/32.)
    velocity[1] = random.uniform(-vSizes[1]/32., vSizes[1]/32.)
    balls.append(BouncingBall(random.randint(minBallSize, maxBallSize),
                              color, position, velocity))
  
  lightDir = np.array([random.uniform(-3.,3.), random.uniform(-3.,3.), 1.])
  lightVel = np.array([random.uniform(-0.001,0.001), random.uniform(-0.001,0.001), 0.])
  videoArray = np.zeros([nFrames, vidSize[1], vidSize[0], 3], dtype=np.uint8)
  for frameCnt in range(nFrames):
    frame = np.zeros([vidSize[1], vidSize[0], 3], dtype=np.uint8)
    for ball in balls:
      drawBall(ball, vSizes, lightDir, frame)
      lightDir += lightVel
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
  def __init__(self, in_features, hidden_layers, out_features, device):
    super(Conv2DRNN, self).__init__()
    self.inf = in_features
    self.hidl = hidden_layers
    self.outf = out_features
    self.device = device
    self.Gates = nn.Conv2d(in_features + hidden_layers,
                           hidden_layers + out_features,
                           KERNEL_SIZE, padding=PADDING).cuda(device)
    self.activation = nn.LeakyReLU(0.1).cuda(device)
    
  def step(self, input_, prev_hidden=None):
    # get batch and spatial sizes
    input_.cuda(self.device)
    
    batch_size = input_.data.size()[0]
    spatial_size = input_.data.size()[2:]
    
    # generate empty prev_state, if None is provided
    if prev_hidden is None:
      state_size = [batch_size, self.hidl] + list(spatial_size)
      prev_hidden = (Variable(torch.zeros(state_size))).cuda(self.device)
    else:
      prev_hidden.cuda(self.device)    
    
    # data size is [batch, channels, height, width]
    stacked_inputs = torch.cat((input_, prev_hidden), 1)
    gates = self.Gates(stacked_inputs)  
    gates = self.activation(gates)
    
    hidden = gates[:, 0:self.hidl, :,:]
    output_layer = gates[:, self.hidl:, :,:]

    return hidden, output_layer
	
  # Inputs here is [batch_size, num_inputs, input_features, height, width]	
  def forward(self, inputs, hidden=None):
    inputs.cuda(self.device)
    if not hidden is None:
      hidden.cuda(self.device)
    steps = inputs.data.size()[1]
    outputs = []

    for i in range(steps):
      input = inputs[:,i,...].squeeze()
      hidden, output = self.step(input, hidden)
      outputs.append(output)
    return hidden, torch.stack(outputs, 1)
    

class VideoNet(nn.Module):
  """
  Generate a network for predicting video
  """
  
  def __init__(self, device, **kwargs):
    super(VideoNet, self).__init__()
    
    self.rnn_layers = nn.ModuleDict()
    self.convT = nn.ModuleDict()
    self.conv = nn.ModuleDict()
    self.device = device
    for i in [3, 6, 12, 24]:
      self.rnn_layers[str(i)] = Conv2DRNN(i, 3, 2*i, device)
      self.convT[str(i)] = nn.ConvTranspose2d(2*i, i, kernel_size=3, stride=2,
                                              padding=PADDING, output_padding=1).cuda(device)
      self.conv[str(i)] = nn.Conv2d(3*i, i, kernel_size=KERNEL_SIZE, padding=PADDING).cuda(device)
      
    self.maxpool = nn.MaxPool2d((2,2)).cuda(device)
    self.dropout = nn.Dropout2d(0.1).cuda(device)

  def forward(self, batch_input):
    loss = nn.MSELoss(reduction='mean')
    rnn_outputs = {}
    batch_input.cuda(self.device)
    layer = batch_input  # layer is [NBatch, NFrames, 3, H, W], H = W = size
    height = batch_input.data.size()[3]
    assert(batch_input.data.size()[4] == height)  # height = width
    assert(2 ** math.log(height, 2) == height)    # height is a power of 2
    
    print("Running forward")
    for i in [3, 6, 12, 24]:
      _, rnn_outputs[i] = self.rnn_layers[str(i)].forward(layer)
	  # rnn_outputs is [NBatch, NFrames, 2*i, H(layer), W(layer)]
      layer = torch.stack([self.dropout(self.maxpool(x)) for x in torch.unbind(rnn_outputs[i], 0)], 0)
	  # layer is [NBatch, NFrames, 2*i, H/2, W/2]
    
    for i in [24, 12, 6, 3]:
      layer = torch.stack([self.dropout(self.convT[str(i)].forward(x)) for x in torch.unbind(layer, 0)], 0)
      # layer is [NBatch, NFrames, i, 2*H(layer), 2*W(layer)]
      layer = torch.cat([rnn_outputs[i], layer], dim=2)
      # layer is [NBatch, NFrames, 3*i, 2*H(layer), 2*W(layer)]
      layer = torch.stack([self.conv[str(i)].forward(x) 
                           for x in torch.unbind(layer, 0)], 0)
      # layer is [NBatch, NFrames, i, 2*H(layer), 2*W(layer)]
	
    if not self.training:
      return loss(batch_input, layer), layer
    
    return loss(batch_input, layer)
	
  
def train(epoch, video_size, model, optimizer_model, use_gpu,
          samples_per_epoch = 100, batch_size=10):
  losses = AverageMeter('Loss', ':6.4f')
  batch_time = AverageMeter('Time', ':6.3f')
  end = time.time()
  model.train()

  print("Craig Alpha: {:d},{:d},{:d}".format(batch_size, NUM_FRAMES, video_size))
  n_iters = int(samples_per_epoch / batch_size)
  print(str(n_iters))
  for iter in range(n_iters):
    video_inputs = np.zeros([batch_size, NUM_FRAMES, 3, video_size, video_size])
    for i in range(batch_size):
      num_balls = random.randint(4,12)
      video_input = buildBouncingBallVideo(num_balls, 
                                           [video_size, video_size], 
                                           NUM_FRAMES)
      video_inputs[i,...] = np.squeeze(np.stack(np.split(video_input, 3, axis=3), 1))
      
    inputs = Variable(torch.from_numpy(video_inputs).float())
    print("Built inputs for iter {}".format(iter))

    loss = model(inputs)
    losses.update(loss, batch_size)

    optimizer_model.zero_grad()
    loss.sum().backward()
    optimizer_model.step()

    #losses[epoch] += loss.data[0]
    batch_time.update(time.time() - end)
    end = time.time()

  if epoch > 0:
    print(epoch, loss.data[0])
    
    
def test(model, video_size, use_gpu, save_output=False):
  num_tests = 100
  batch_size = 15
  
  batch_time = AverageMeter('Time', ':6.3f')
  losses = AverageMeter('Loss', ':6.4f')
  progress = ProgressMeter(num_tests, 'Test: ', batch_time, losses)

  model.eval()
  with torch.no_grad():
    end = time.time()
    print('T {:6.3f}'.format(end))
    for iter in range(num_tests):
      video_inputs = np.zeros([batch_size, NUM_FRAMES, 3, video_size, video_size])
      for i in range(batch_size):
        num_balls = random.randint(4,12)
        video_input = buildBouncingBallVideo(num_balls, 
                                             [video_size, video_size], 
                                             NUM_FRAMES)
        video_inputs[i,...] = np.squeeze(np.stack(np.split(video_input, 3, axis=3), 1))
        
      inputs = Variable(torch.from_numpy(video_inputs).float())

      loss, outputs = model(inputs)
      losses.update(loss, batch_size)
      print('L0: {}'.format(loss))
      print('L {:d}: {:6.4f}'.format(iter, loss))
      
      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()
      
      if iter % 20 == 0:
        progress.printb(iter)
        if iter == 0 and save_output:
          diffs = np.squeeze(((outputs.data).cpu().numpy())[-1,...] - video_inputs[-1,...])
          print('DS: {}'.format(diffs.shape()))
          video_output = np.squeeze(np.stack(np.split(np.abs(diffs), axis=1), 4))
          vidio.vwrite('output_diff.mp4', video_output)
          
  return losses.avg

  
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
    out_video = buildBouncingBallVideo(20, [args.size, args.size], NUM_FRAMES)
    vidio.vwrite("test_video.mp4", out_video)
    return
  
  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
  use_gpu = torch.cuda.is_available()
  if args.use_cpu: use_gpu = False
  device = torch.device('cuda:0' if use_gpu else 'cpu')

  if not args.evaluate:
    print(osp.join(args.save_dir, 'log_train.txt'))
  else:
    print(osp.join(args.save_dir, 'log_test.txt'))
  print("==========\nArgs:{}\n==========".format(args))

  if use_gpu:
    print("Currently using GPU {}".format(args.gpu_devices))
    cudnn.benchmark = True
  else:
    print("Currently using CPU (GPU is highly recommended)")
	
  print("Initializing model")
  model = VideoNet(device)
  print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))	
  
  start_epoch = args.start_epoch

  optimizer_model = optim.Adam(model.parameters(), lr=args.lr)
  print("Optimizer set up")
  
  if args.resume:
    print("Loading checkpoint from '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['state_dict'])
    start_epoch = checkpoint['epoch']

  if use_gpu:
    print("Using GPU")
    model = nn.DataParallel(model).cuda()
  else:
    print("Not using GPU")

  if args.evaluate:
    print("Evaluate only")
    test(model, args.size, use_gpu)
    return

  start_time = time.time()
  train_time = 0
  best_rank1 = -np.inf
  best_epoch = 0
  print("==> Start training")

  for epoch in range(start_epoch, args.max_epoch):
    start_train_time = time.time()
    train(epoch, args.size, model, optimizer_model, use_gpu)
    train_time += time.time() - start_train_time
    print("Trained epoch {}".format(epoch))
        
    if (epoch+1) % 1 == 0 or (epoch+1) == args.max_epoch:
      print("==> Test: {}".format(epoch))
      rank1 = test(model, args.size, use_gpu, True)
      is_best = rank1 > best_rank1
      if is_best:
        best_rank1 = rank1
        best_epoch = epoch + 1

      if use_gpu:
        state_dict = model.module.state_dict()
      else:
        state_dict = model.state_dict()
      torch.save({ 'state_dict': state_dict, 'rank1': rank1, 'epoch': epoch, }, is_best, 
                 osp.join(args.save_dir, 'checkpoint_ep' + str(epoch+1) + '.pth.tar'))

    print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = datetime.timedelta(seconds=elapsed)
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, datetime.timedelta(seconds=train_time)))
    
    
if __name__ == '__main__':
  main()