import cv2, numpy as np, random, math
import skvideo
#skvideo.setFFmpegPath('/usr/local/lib/python2.7/dist-packages/ffmpeg/')
import skvideo.io as vidio

NUM_DIMS = 3

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

 