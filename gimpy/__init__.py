"""
    Gimpy module
    ============

    Provides pure Python versions of GIMP operations.

    Well, right now there is only one algorithm...

    :copyright: 2013 by David Volquartz Lebech
    :license: MIT, see LICENSE for details.

"""
import colorsys

def scale(val, oldmin, oldmax, newmin, newmax):
    """Helper function for scaling a value to a different range."""
    oldrange = oldmax - oldmin
    newrange = newmax - newmin
    newval= (((val - oldmin) * newrange) / oldrange) + newmin
    return newval

def rgb_to_hls(r, g, b):
    """Convert a rgb color value into a hls color value.

    This function works with 0-255 color values.

    :param r:
        Red value
    :param g:
        Green value
    :param b:
        Blue value

    """
    r = scale(r, 0, 255, 0.0, 1.0)
    g = scale(g, 0, 255, 0.0, 1.0)
    b = scale(b, 0, 255, 0.0, 1.0)
    hls = colorsys.rgb_to_hls(r, g, b)
    h = scale(hls[0], 0.0, 1.0, 0, 255)
    l = scale(hls[1], 0.0, 1.0, 0, 255)
    s = scale(hls[2], 0.0, 1.0, 0, 255)
    return [h, l, s]

def hls_to_rgb(h, l, s):
    """Convert a hls color value into a rgb color value.

    This function works with 0-255 color values.

    :param h:
        Hue value
    :param l:
        Light/luminance value
    :param s:
        Saturation value

    """
    h = scale(h, 0, 255, 0.0, 1.0)
    l = scale(l, 0, 255, 0.0, 1.0)
    s = scale(s, 0, 255, 0.0, 1.0)
    rgb = colorsys.hls_to_rgb(h, l, s)
    r = scale(rgb[0], 0.0, 1.0, 0, 255)
    g = scale(rgb[1], 0.0, 1.0, 0, 255)
    b = scale(rgb[2], 0.0, 1.0, 0, 255)
    return [r, g, b]
