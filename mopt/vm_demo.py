import time

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


class Point(object):

  def __init__(self, ax, point):
    self.point = point
    self.press = None
    self.lines = []

    self.ax = ax
    self.ax.add_patch(point)

    self.cidpress = None
    self.cidmotion = None
    self.cidrelease = None

  @property
  def pos(self):
    return self.point.center

  @pos.setter
  def pos(self, position):
    self.point.center = (position[0], position[1])
    # self.point.center = (self.point.center[0], position[1])
    for line in self.lines:
      (x1, x2), (y1, y2) = line.get_data()
      if self == line.points[0]:
        line.set_data((self.pos[0], x2), (self.pos[1], y2))
      elif self == line.points[1]:
        line.set_data((x1, self.pos[0]), (y1, self.pos[1]))

  def on_press(self, event):
    self.press = self.pos, event.xdata, event.ydata
    self.point.set_animated(True)

  def on_motion(self, event):
    self.pos, xpress, ypress = self.press
    dx = event.xdata - xpress
    dy = event.ydata - ypress
    self.pos = (self.pos[0] + dx, self.pos[1] + dy)

  def on_release(self, event):
    self.press = None
    self.point.set_animated(False)

  def disconnect(self):
    'disconnect all the stored connection ids'
    self.point.figure.canvas.mpl_disconnect(self.cidpress)
    self.point.figure.canvas.mpl_disconnect(self.cidmotion)
    self.point.figure.canvas.mpl_disconnect(self.cidrelease)


class VMPoint(Point):

  @property
  def pos(self):
    return self.point.center

  @pos.setter
  def pos(self, position):
    self.point.center = (position[0], position[1])


class Front(object):

  def __init__(self, ax, vm_point, points):
    self.vm_point = vm_point
    self.points = points
    self.lines = []
    for i in range(1, len(self.points)):
      line = Line2D([self.points[i].pos[0], self.points[i - 1].pos[0]],
                    [self.points[i].pos[1], self.points[i - 1].pos[1]],
                    color='k', zorder=0)
      ax.add_line(line)
      self.lines.append(line)
      self.points[i].lines.append(line)
      self.points[i - 1].lines.append(line)
      line.points = (self.points[i], self.points[i - 1])

    self.lock = None
    self.background = [None, None]
    self.canvas = (self.vm_point.point.figure.canvas, self.points[0].point.figure.canvas)
    self.axes = (self.vm_point.point.axes, self.points[0].point.axes)

    self.register_handlers()

  def register_handlers(self):
    self.vm_point.disconnect()
    self.vm_point.cidpress = vm_point.point.figure.canvas.mpl_connect('button_press_event', self.on_press)
    self.vm_point.cidmotion = vm_point.point.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
    self.vm_point.cidrelease = vm_point.point.figure.canvas.mpl_connect('button_release_event', self.on_release)
    for point in self.points:
      point.disconnect()
      point.cidpress = point.point.figure.canvas.mpl_connect('button_press_event', self.on_press)
      point.cidmotion = point.point.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)
      point.cidrelease = point.point.figure.canvas.mpl_connect('button_release_event', self.on_release)

  def redraw(self, full=False, track=False):
    # draw everything but the selected rectangle and store the pixel buffer
    if full:
      self.canvas[0].draw()
      if not track:
        self.background[0] = self.canvas[0].copy_from_bbox(self.axes[0].bbox)
      self.canvas[1].draw()
      self.background[1] = self.canvas[1].copy_from_bbox(self.axes[1].bbox)
    else:
      # restore the background region
      if self.background[0] is not None:
        self.canvas[0].restore_region(self.background[0])
      if self.background[1] is not None:
        self.canvas[1].restore_region(self.background[1])
    self.axes[0].draw_artist(self.vm_point.point)
    self.canvas[0].blit(self.axes[0].bbox)
    map(lambda line: self.axes[1].draw_artist(line), self.lines)
    map(lambda point: self.axes[1].draw_artist(point.point), self.points)
    self.canvas[1].blit(self.axes[1].bbox)

  def on_press(self, event):
    if self.lock is not None:
      return

    track = False
    if event.inaxes == self.vm_point.point.axes and self.vm_point.point.contains(event)[0]:
      self.lock = self.vm_point
    else:
      for point in self.points:
        if event.inaxes == point.point.axes and point.point.contains(event)[0]:
          self.lock = point
          track = True
          break
    if self.lock is None:
      return

    self.vm_point.on_press(event)
    map(lambda point: point.on_press(event), self.points)
    map(lambda line: line.set_animated(True), self.lines)
    self.redraw(True, track=track)

  def on_motion(self, event):
    if event.inaxes == self.axes[0] and self.lock is self.vm_point:
      self.vm_point.on_motion(event)
      vmx, vmy = self.vm_point.pos
      self.points[0].pos = (0, 1 - vmx)
      self.points[1].pos = (1, 1)
      self.points[2].pos = (2, 1 + vmy)
      self.redraw(full=False)
    else:
      for point in self.points:
        if event.inaxes == self.axes[1] and self.lock is point:
          point.on_motion(event)
          x0, y0 = self.points[0].pos
          x1, y1 = self.points[1].pos
          x2, y2 = self.points[2].pos
          self.vm_point.pos = ((y1 - y0) / abs(x1 - x0), (y2 - y1) / abs(x2 - x1))
          self.redraw(full=False)
          break

  def on_release(self, event):
    self.lock = None
    self.background = [None, None]
    self.vm_point.on_release(event)
    map(lambda point: point.on_release(event), self.points)
    map(lambda line: line.set_animated(False), self.lines)
    self.canvas[0].draw()
    self.canvas[1].draw()


# %matplotlib tk
plt.rc('figure', figsize=(10, 5))

fig = plt.figure()

ax = fig.add_subplot(121)
ax.plot([-1, 1, 1, -1, -1], [1, 1, -1, -1, 1], alpha=0.5)
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.grid()
vm_point = VMPoint(ax, patches.Circle((0.0, 0.0), 0.1, fc='k', alpha=1))

ax = fig.add_subplot(122)
plt.xlim([-0.1, 2.1])
plt.ylim([0, 2])
plt.grid()
point1 = Point(ax, patches.Circle((0, 1), 0.05, fc='r', alpha=1))
point2 = Point(ax, patches.Circle((1, 1), 0.05, fc='g', alpha=1))
point3 = Point(ax, patches.Circle((2, 1), 0.05, fc='b', alpha=1))
front = Front(ax, vm_point, [point1, point2, point3])

plt.show()
