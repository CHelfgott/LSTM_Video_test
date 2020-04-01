# -*- coding: utf-8 -*-
"""
Created on Tue May 22 14:07:01 2018
Simple video prediction task -- NN test
@author: Craig
"""

from bouncing_ball_utils import buildBouncingBallVideo
from convlstm import ConvLSTM
from convrnn import Conv2DRNN, Conv2DRNNCell
import cv2, numpy as np, os, random, math
import os.path as osp
import time
import datetime
import skvideo
#skvideo.setFFmpegPath('/usr/local/lib/python2.7/dist-packages/ffmpeg/')
import skvideo.io as vidio
import torch, torch.nn as nn, torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import argparse
import shutil

INPUT_SIZE = 128
NUM_FRAMES = 100
NUM_EPOCHS = 100

KERNEL_SIZE = 3
PADDING = KERNEL_SIZE // 2


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

class VideoNet(nn.Module):
  """
  Generate a network for predicting video
  """
  
  def __init__(self, device, debug=False, **kwargs):
    super(VideoNet, self).__init__()
    
    self.rnn_layers = nn.ModuleDict()
    self.convT = nn.ModuleDict()
    self.conv = nn.ModuleDict()
    self.device = device
    self.debug = debug
    for i in [3, 6, 12, 24]:
      self.rnn_layers[str(i)] = Conv2DRNNCell(i, 3, 2*i, kernel_size=(KERNEL_SIZE, KERNEL_SIZE),
                                              padding=(PADDING, PADDING))
      self.convT[str(i)] = nn.ConvTranspose2d(2*i, i, kernel_size=3, stride=2,
                                              padding=PADDING, output_padding=1)
      self.conv[str(i)] = nn.Conv2d(3*i, i, kernel_size=KERNEL_SIZE, padding=PADDING)
      
    self.maxpool = nn.MaxPool2d((2,2))
    self.dropout = nn.Dropout2d(0.1)

  def forward(self, batch_input, debug=False):
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
      if self.debug:
        print(self.rnn_layers[str(i)].weight.data)
    
    for i in [24, 12, 6, 3]:
      layer = torch.stack([self.dropout(self.convT[str(i)].forward(x)) for x in torch.unbind(layer, 0)], 0)
      # layer is [NBatch, NFrames, i, 2*H(layer), 2*W(layer)]
      layer = torch.cat([rnn_outputs[i], layer], dim=2)
      # layer is [NBatch, NFrames, 3*i, 2*H(layer), 2*W(layer)]
      layer = torch.stack([self.conv[str(i)].forward(x) 
                           for x in torch.unbind(layer, 0)], 0)
      # layer is [NBatch, NFrames, i, 2*H(layer), 2*W(layer)]
      if self.debug:
        print(self.convT[str(i)].weight.data)
        
    if not self.training:
      return loss(batch_input, layer), layer
    
    return loss(batch_input, layer)
	
  
def train(epoch, video_size, model, optimizer_model,
          samples_per_epoch = 1000, batch_size=10):
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

    data = torch.from_numpy(video_inputs).float()
    if torch.cuda.is_available():
      data = data.cuda()
    inputs = Variable(data)
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
    print(epoch, loss.item())
    
    
def test(model, video_size, save_output=False):
  num_tests = 100
  batch_size = 15
  
  batch_time = AverageMeter('Time', ':6.3f')
  losses = AverageMeter('Loss', ':6.4f')
  progress = ProgressMeter(num_tests, 'Test: ', batch_time, losses)

  model.eval()
  video_output = None
  with torch.no_grad():
    end = time.time()
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
      
      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()
      
      if iter % 20 == 0:
        progress.printb(iter)
      if iter == num_tests - 1 and save_output:
        diffs = np.squeeze(((outputs.data).cpu().numpy())[-1,...] - video_inputs[-1,...])
        video_output = np.squeeze(np.stack(np.split(np.abs(diffs), 3, axis=1), 4)).astype(int)
        print(video_output.shape)
          
  return losses.avg, video_output

def evaluate(model, video_file):
  model.eval()
  with open(video_file) as vf:
    video_inputs = np.expand_dims(np.transpose(vidio.vread(vf),(0,3,1,2)), axis=0)
  inputs = Variable(torch.from_numpy(video_inputs).float())
  
  loss, outputs = model(inputs)
  diffs = np.squeeze(((outputs.data).cpu().numpy())[-1,...] - video_inputs[-1,...])
  video_output = np.squeeze(np.stack(np.split(np.abs(diffs), 3, axis=1), 4)).astype(int)

  return loss, video_output
  
def main():
  parser = argparse.ArgumentParser(description='Train video prediction model')
  parser.add_argument('--just_video', action='store_true',
                      help="flag to just generate one video")
  parser.add_argument('--debug', action='store_true',
                      help="outputs weights in training for debug purposes")
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
  parser.add_argument('--test', action='store_true', help="test evaluation only")
  parser.add_argument('--evaluate', type=str, default='',
                      help="evaluate on an input video")
  parser.add_argument('--save-dir', type=str, default='logs')
  parser.add_argument('--use-cpu', action='store_true', help="use cpu")
  parser.add_argument('--gpu-devices', default='0', type=str,
                      help='gpu device ids for CUDA_VISIBLE_DEVICES')
  args = parser.parse_args()
  
  if args.just_video:
    out_video = buildBouncingBallVideo(20, [args.size, args.size], NUM_FRAMES)
    print(out_video.shape)
    vidio.vwrite("test_video.mp4", out_video)
    return
  
  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
  use_gpu = torch.cuda.is_available()
  if args.use_cpu: use_gpu = False
  device = torch.device('cuda:0' if use_gpu else 'cpu')

  if not args.test:
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
  model = VideoNet(device, args.debug)
  print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))
  print(model)
  
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

  if args.test:
    print("Evaluate test score only")
    test(model, args.size)
    return
    
  if args.evaluate:
    print("Evaluate performance on input video")
    score, video_output = evaluate(model, args.evaluate)
    print("Score = " + str(score))
    if not video_output is None:
      vidio.vwrite('evaluate_diff.mp4', video_output)
    return  

  start_time = time.time()
  train_time = 0
  best_rank1 = -np.inf
  best_epoch = 0
  print("==> Start training")

  for epoch in range(start_epoch, args.max_epoch):
    start_train_time = time.time()
    train(epoch, args.size, model, optimizer_model)
    train_time += time.time() - start_train_time
    print("Trained epoch {}".format(epoch))
        
    if (epoch+1) % 5 == 0 or (epoch+1) == args.max_epoch:
      print("==> Test: {}".format(epoch))
      rank1, video_output = test(model, args.size, True)
      if not video_output is None:
        vidio.vwrite('output_diff{:d}.mp4'.format(epoch), video_output)
      is_best = rank1 > best_rank1
      if is_best:
        best_rank1 = rank1
        best_epoch = epoch + 1

      if use_gpu:
        state_dict = model.module.state_dict()
      else:
        state_dict = model.state_dict()
      save_path = osp.join(args.save_dir, 'checkpoint_ep{:d}.pth.tar'.format(epoch+1))
      torch.save({ 'state_dict': state_dict, 'rank1': rank1, 'epoch': epoch }, save_path)
      if is_best:
        shutil.copy(save_path, osp.join(args.save_dir, 'model-best.pth.tar'))

    print("==> Best Rank-1 {:.1%}, achieved at epoch {}".format(best_rank1, best_epoch))

    elapsed = round(time.time() - start_time)
    elapsed = datetime.timedelta(seconds=elapsed)
    print("Finished. Total elapsed time (h:m:s): {}. Training time (h:m:s): {}.".format(elapsed, datetime.timedelta(seconds=train_time)))
    
    
if __name__ == '__main__':
  main()
