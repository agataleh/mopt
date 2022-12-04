"""
A collection of algorithms related to space-filling curves.

# The Curves

The following traversals of all points in a space are supported (some are true
space-filling curves, some are not):

- __hilbert__:    Hilbert curve
- __natural__:    A natural-order traversal of all points, where each co-ordinate is simply treated as a digit.
- __zigzag__:     A traversal of all points that zig-zags to ensure that each point differs from the previous point by a unit-offset in only one dimension.
- __zorder__:     Z-order curve


# More info

Development on Scurve is usually spurred along by posts on my blog. Some of
scurve's features are documented and illustrated in the following posts:

- [Portrait of the Hilbert Curve](http://corte.si/posts/code/hilbert/portrait/index.html)
- [Generating colour maps with space-filling curves](http://corte.si/posts/code/hilbert/swatches/index.html)
- [Hilbert Curve + Sorting Algorithms + Procrastination = ?](http://corte.si/posts/code/sortvis-fruitsalad/index.html)
"""

from . import graycurve, hcurve, hilbert, natural, zigzag, zorder

curveMap = {
    "hcurve": hcurve.Hcurve,
    "hilbert": hilbert.Hilbert,
    "zigzag": zigzag.ZigZag,
    "zorder": zorder.ZOrder,
    "natural": natural.Natural,
    "gray": graycurve.GrayCurve,
}
curves = curveMap.keys()


def fromSize(curve, dimension, size):
  """
      A convenience function for creating a specified curve by specifying
      size and dimension. All curves implement this common interface.
  """
  return curveMap[curve].fromSize(dimension, size)


def fromOrder(curve, dimension, order):
  """
      A convenience function for creating a specified curve by specifying
      order and dimension. All curves implement this common interface, but
      the meaning of "order" may differ for each curve.
  """
  return curveMap[curve](dimension, order)
