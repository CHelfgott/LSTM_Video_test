# -*- coding: utf-8 -*-
"""
Created on Tue May 22 14:07:01 2018
Simple video prediction task -- NN test
@author: Craig
"""

import cv2, numpy as np, os, random, math, glob
import skvideo.io as vidio
import tensorflow as tf
import argparse

INPUT_SIZE = 512
NUM_FRAMES = 50
NUM_EPOCHS = 100

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


activations = {
    'elu' : tf.nn.elu,
    'relu' : tf.nn.relu,
    'lrelu' : tf.nn.leaky_relu
}


def conv2d(input, filters, name, **kwargs):
  activation = None
  if 'activation' in kwargs:
    activation = activations[kwargs['activation']]
    del kwargs['activation']

#  ci_shape = [tf.shape(input)[0] * tf.shape(input)[1]] + tf.shape(input)[2:]
#  conv_input = tf.reshape(input, ci_shape)
  conv = tf.layers.conv2d(input, filters, name=name, **kwargs)
#  co_shape = [tf.shape(input)[0], tf.shape(input)[1]] + tf.shape(conv)[1:]

  if activation:
    batch_norm = tf.layers.batch_normalization(conv)
    return activation(batch_norm)
#    return tf.reshape(activation(batch_norm), co_shape)
  else:
#    return tf.reshape(conv, co_shape)
    return conv
  
  
def conv2dLSTM(input_shape, filters, name):
  activation = tf.nn.leaky_relu
  kernel_shape = [3,3]

  cell = tf.contrib.rnn.Conv2DLSTMCell(input_shape=input_shape, 
                                       kernel_shape=kernel_shape, 
                                       output_channels=filters,
                                       name=name)

  return cell
  if activation:
    batch_norm = tf.layers.batch_normalization(cell)
    return activation(batch_norm)
  else:
    return cell


def pool2d(input, name, **kwargs):
#  pi_shape = [tf.shape(input)[0] * tf.shape(input)[1]] + tf.shape(input)[2:]
#  pool_input = tf.reshape(input, pi_shape)
  pool = tf.layers.max_pooling2d(inputs=input, name=name, **kwargs)
#  po_shape = [tf.shape(input)[0], tf.shape(input)[1]] + tf.shape(pool)[1:]
  return pool 
#  return tf.reshape(pool, po_shape)


def build_forward(size, nframes, mode, lr=0.0001):
  assert(2 ** math.log(size, 2) == size) # size is a base 2 power
  tf.reset_default_graph()
  input_layer = tf.placeholder(tf.float32, shape=(nframes, size, size, 3),\
                               name='input')

#  activation = tf.nn.leaky_relu

  kinit = tf.contrib.layers.xavier_initializer()

  conv_args = {
    'kernel_size' : [3,3],
    'padding' : 'same',
    'activation' : 'lrelu',
    'kernel_initializer' : kinit
  }

  downsamples = {}
  
#  with tf.variable_scope('first_layer'):
  conv1 = conv2d(input_layer, 32, 'conv1', **conv_args)
  cell1 = conv2dLSTM([size,size,32], 32, 'conv2')
  cell1_inputs = tf.unstack(tf.expand_dims(conv1, axis=0), axis=1)
  lstm_conv1, state1 = tf.nn.static_rnn(cell1, cell1_inputs, dtype=tf.float32)
  conv2 = tf.stack(lstm_conv1, axis=0)

  in_layer = tf.concat([conv1, tf.squeeze(conv2)], axis=-1)

  downsamples[size] = in_layer

  pool_args = {
    'pool_size' : [2,2],
    'strides' : 2
  }
  
  # Downsampling
  csize = size
  for filters in [64, 128, 256]:
    csize /= 2
    with tf.variable_scope('down_sample_%d' % filters):
      pool = pool2d(in_layer, name='down_maxpool_%d' % filters, **pool_args)
      conv = conv2d(pool, filters, 'down1_%d' % filters, **conv_args)
      cell = conv2dLSTM([csize,csize,filters], filters, 'down2_%d' % filters)
      cell_inputs = tf.unstack(tf.expand_dims(conv, axis=0), axis=1)
      lstm_conv, state = tf.nn.static_rnn(cell, cell_inputs, dtype=tf.float32)
      in_layer = tf.concat([conv, tf.squeeze(tf.stack(lstm_conv, axis=0))], axis=-1)
      downsamples[csize] = in_layer
                 
  # Upsampling
  for filters in [128, 64, 32]:
    csize *= 2
    with tf.variable_scope('upsample_%d' % filters):
      upsample = tf.layers.conv2d_transpose(in_layer, filters, 2, strides=[2,2])
      concat = tf.concat([downsamples[csize], upsample], axis=-1)
      conv2 = conv2d(concat, filters, 'up1_%d' % filters, **conv_args)
      cell = conv2dLSTM([csize,csize,filters], filters, 'up2_%d' % filters)
      cell_inputs = tf.unstack(tf.expand_dims(conv2, axis=0), axis=1)
      lstm_conv2, state = tf.nn.static_rnn(cell, cell_inputs, dtype=tf.float32)      
      in_layer = tf.concat([conv2, tf.squeeze(tf.stack(lstm_conv2, axis=0))], axis=-1)
  
  output_layer = conv2d(in_layer, 3, 'output', **conv_args)
  
  if mode == tf.estimator.ModeKeys.PREDICT:
    return input_layer, output_layer
  else:
    assert mode == tf.estimator.ModeKeys.TRAIN
    
  # L2 regularization -- not actually used.
  tvars = tf.trainable_variables()
  lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in tvars 
                     if 'bias' not in v.name ]) * 0.001

  loss = tf.losses.mean_squared_error(input_layer[-1, 1:, ...],
                                      output_layer[-1, :-1, ...])

  optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1e-4)  
  
  gradients = tf.gradients(loss, tf.trainable_variables())
  gradients = list(zip(gradients, tf.trainable_variables()))

  # Plot gradients in tensorboard
  for grad_var_pair in gradients:
    current_variable = grad_var_pair[1]
    current_gradient = grad_var_pair[0]
    gradient_name_to_save = current_variable.name.replace(":", "_")
    if current_gradient is not None:
      tf.summary.histogram(gradient_name_to_save, current_gradient) 

  train_op = optimizer.apply_gradients(grads_and_vars=gradients)
  
  tf.summary.scalar('loss', loss)
  return train_op, input_layer, loss, output_layer


def train(start_iter, end_iter, checkpoint=None):
  samples_per_epoch = 1000
  
  train_op, in_place, loss_t, output_video_t = \
      build_forward(INPUT_SIZE, NUM_FRAMES, tf.estimator.ModeKeys.TRAIN)
      
  init = tf.global_variables_initializer()
  saver = tf.train.Saver()
  
  step = 0
  best_loss = 1e10
  merged = tf.summary.merge_all()
  with tf.Session() as sess:
    if checkpoint:
      try:
        saver.restore(sess, 'weights/{}_checkpoint'.format(checkpoint))
      except:
        print(checkpoint + ' is not a valid checkpoint to restore from.')
        sess.run(init)
    else:
      sess.run(init)
      checkpoint = 'log'
        
    writer = tf.summary.FileWriter('./logs/gpu_' + os.environ['CUDA_VISIBLE_DEVICES'], sess.graph)
    for epoch in range(start_iter, end_iter):
      epoch_loss = 0.0
      for i in range(samples_per_epoch):
        num_balls = random.randint(4,12)
        video_input = buildBouncingBallVideo(num_balls, 
                                             [INPUT_SIZE, INPUT_SIZE], 
                                             NUM_FRAMES)
        summary, loss, _ = sess.run([merged, loss_t, train_op],
                                    { in_place : video_input, })  
        if step % 20 == 0:
          writer.add_summary(summary, step)
    
        step += 1
                
        print(' %d [%d/%d] - Loss: %f' % (epoch, i, samples_per_epoch, loss))
                
        epoch_loss += loss
      # remember best prec@1 and save checkpoint
      saver.save(sess, 'weights/{}_checkpoint'.format(checkpoint))
      if epoch_loss < best_loss:
        best_loss = epoch_loss
        saver.save(sess, 'weights/{}_best'.format(checkpoint))
        
        
def predict(data_dir, output_dir, checkpoint, batch_size):
    in_place, output_video_t = build_forward(INPUT_SIZE, NUM_FRAMES,
                                             tf.estimator.ModeKeys.PREDICT)

    saver = tf.train.Saver()
    with tf.Session() as sess:
      try:
        saver.restore(sess, 'weights/{}_best'.format(checkpoint))
      except Exception as e:
        print('{}: {} is not a valid checkpoint to restore from.'.format(e, checkpoint))

      prediction_folder = os.path.join(output_dir, 'output')
      # check that prediction folder exists
      if not os.path.exists(prediction_folder):
        os.mkdir(prediction_folder) 
        
      input_videos = glob.glob(os.path.join(data_dir, '/*.mp4'))
      
      for ivf in input_videos:
        in_video = vidio.vread(ivf)
        out_video = sess.run([output_video_t],
            { in_place : in_video[:NUM_FRAMES, :INPUT_SIZE, :INPUT_SIZE] })
        base_filename = ivf.split('/')[-1]
        out_filename = os.path.join(prediction_folder, base_filename)
        writer = vidio.FFmpegWriter(out_filename)
        for i in range(out_video.shape[0]):
          writer.writeFrame(out_video[i,...])
        writer.close()
          

if __name__ == '__main__':
#  video = buildBouncingBallVideo(12, [640,480], 500)
#  writer = vidio.FFmpegWriter('outputvideo.mp4')
#  for i in range(100):
#    writer.writeFrame(video[i,...])
#  writer.close()  
  parser = argparse.ArgumentParser()
  parser.add_argument('--gpu', default='0', help='Determines whether or not to run on GPU')
  parser.add_argument('--logname', default='log', type=str, help='Root name for session checkpoints')
  parser.add_argument('--lr', default=0.0001, type=float, help='Learning rate')
  parser.add_argument('--momentum', default=0.9, type=float, help='SGD Momentum')
  parser.add_argument('--weight_decay', default=0.0005, type=float, help='Weight Decay')
  parser.add_argument('--data_dir', default='../data')
  parser.add_argument('--output_dir', default='../output')
  parser.add_argument('-pr', '--predict', dest='predict', action='store_true',
                      help='generate prediction masks')
  parser.add_argument('-tr', '--train', dest='train', action='store_true',
                      help='train road detection NN weights')
  parser.add_argument('--resume_at_epoch', default=0, type=int, help='Resume training from where it left off')
  args = parser.parse_args()

  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
              
  if args.train == args.predict:
    print('Must either train or predict, not both or neither.')
  elif args.train:
    if args.resume_at_epoch > 0:
      train(args.resume_at_epoch, 100, args.logname)
    else:
      train(0, 100)
  else:
    # In this case, predict.
    predict(args.data_dir, args.output_dir, args.logname)
