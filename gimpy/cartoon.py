"""
    cartoon
    =======

    A filter for creating a cartoon like effect on an image.

    :copyright: 2013 by David Volquartz Lebech
    :license: MIT, see LICENSE for details.

"""
import logging

from PIL import ImageFilter

from gimpy import rgb_to_hls, hls_to_rgb

def cartoon(img):
    """Cartoonifies the given image.

    :param img:
        The PIL image to filter

    """
    # Preprocessing of image.
    # Creates a Gaussian blurred and Mean filtered version of the image.
    # These are used in the cartoon algorithm.
    blur = img.filter(ImageFilter.GaussianBlur()).load()
    avg =  img.filter(ImageFilter.Kernel((5,5), [1.0/25.0]*25)).load()
    temp_img = img.copy()
    width, height = temp_img.size

    img_px = temp_img.load()

    # Algorithm (copied from GIMP plug-ins/common/cartoon.c)
    # For each pixel, calculate pixel intensity value to be: avg (blur radius)
    # relative diff = pixel intensity / avg (mask radius)
    # If relative diff < Threshold
    #   intensity mult = (Ramp - MIN (Ramp, (Threshold - relative diff))) / Ramp
    #   pixel intensity *= intensity mult
    ramp = 0.05
    threshold = 1.0
    for col in range(0, width):
        for row in range(0, height):
            rgb = blur[col, row]
            hls = rgb_to_hls(rgb[0], rgb[1], rgb[2])
            blur_light = hls[1]

            rgb = avg[col, row]
            hls = rgb_to_hls(rgb[0], rgb[1], rgb[2])
            avg_light = hls[1]

            rgb = img_px[col, row]
            hls = rgb_to_hls(rgb[0], rgb[1], rgb[2])
            img_light = hls[1]

            mult = 0.0
            if avg_light != 0:
                diff = blur_light / avg_light
                if diff < threshold:
                    if ramp == 0.0:
                        mult = 0.0
                    else:
                        mult = (ramp - min(ramp, threshold-diff)) / ramp
                else:
                    mult = 1.0

            img_light = max(0, min(blur_light*mult, 255))

            rgb = hls_to_rgb(hls[0], img_light, hls[2])
            img_px[col, row] = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
            
    return temp_img

"""
My miserable attempt to exactly translate the the algorthm from C to Python can
be seen below.
C skills... fail.

The algorithm runs but it results in a 100% lighted image. In other words, the
resulting image is near white. It is also much much slower than the above
version.

After playing around with the algorithm for a long time, I gave up and applied
the basic principles of the algorithm and simplified it.
"""

from math import sqrt, log, pi, exp, sin, cos

cvals = {
        'threshold': 1.0,
        'mask_radius': 7.0,
        'pct_black': 0.2
        }

def compute_ramp(blur, avg, width, height, pct_black):
    hist = [0]*100
    count = 0;

    for col in range(0, width):
        for row in range(0, height):
            blur_light = blur[col][row]
            avg_light = avg[col][row]

            if avg_light != 0:
                diff = blur_light / avg_light
                if diff < 1.0:
                    hist[int(diff*100)] += 1
                    count += 1

    if (pct_black == 0.0 or count == 0):
        return 1.0

    s = 0;
    for i in range(0, 101):
        s += hist[i]
        if s/count > pct_black:
            return 1.0 - (i / 100.0)
    return 0.0

def find_constants(n_p, n_m, d_p, d_m, bd_p, bd_m, std_dev):
    """Returns the constants used in the implemenation of a casual sequence
    using a 4th order approximation of the gaussian operator
    
    """
    c = [None]*8  # Constants

    # The normal (or gaussian) distribution is defined as:
    # f(x) = 1 / std_dev*sqrt(2*pi) *           term1
    #        exp(-(x-mean)^2/2*std_dev^2)       term2
    # Wow, that's ugly. Take a look at the Wikipedia article instead:
    # http://en.wikipedia.org/wiki/Normal_distribution

    # Calculate the denominator of term1
    div = sqrt(2*pi) * std_dev

    # Put in some constants.
    # TODO, understand these numbers
    c[0] = -1.783  / std_dev
    c[1] = -1.723  / std_dev
    c[2] =  0.6318 / std_dev
    c[3] =  1.997  / std_dev
    c[4] =  1.6803 / div
    c[5] =  3.735  / div
    c[6] = -0.6803 / div
    c[7] = -0.2598 / div

    # Calculate a whole bunch of weird numbers.
    # TODO, understand these numbers.
    n_p [0] = c[4] + c[6];
    n_p [1] = (exp(c[1]) * (c[7] * sin(c[3]) - (c[6] + 2 * c[4]) * cos(c[3])) +
               exp(c[0]) * (c[5] * sin(c[2]) - (2 * c[6] + c[4]) * cos(c[2])))
    n_p [2] = (2 * exp(c[0] + c[1]) * 
               ((c[4] + c[6]) * cos(c[3]) * cos(c[2]) -
               c[5] * cos(c[3]) * sin(c[2]) - c[7] * cos(c[2]) * sin(c[3])) + 
               c[6] * exp(2*c[0]) + c[4] * exp(2*c[1]))
    n_p [3] = (exp(c[1] + 2 * c[0]) * (c[7] * sin(c[3]) - c[6] * cos(c[3])) +
               exp(c[0] + 2 * c[1]) * (c[5] * sin(c[2]) - c[4] * cos(c[2])))
    n_p [4] = 0.0
  
    d_p [0] = 0.0
    d_p [1] = -2 * exp(c[1]) * cos(c[3]) - 2 * exp(c[0]) * cos(c[2])
    d_p [2] = (4 * cos(c[3]) * cos(c[2]) * exp(c[0] + c[1]) + exp(2 * c[1]) +
              exp(2*c[0]))
    d_p [3] = (-2 * cos(c[2]) * exp(c[0]+2*c[1]) - 2 * cos(c[3]) * 
               exp(c[1]+2*c[0]))
    d_p [4] = exp(2 * c[0] + 2 * c[1])

    # Copy list
    d_m = list(d_p)

    n_m[0] = 0.0
    for i in range(1, 5):
        n_m [i] = n_p[i] - d_p[i] * n_p[0]
  
    sum_n_p = 0.0
    sum_n_m = 0.0
    sum_d   = 0.0

    for i in range(0,5):
        sum_n_p += n_p[i]
        sum_n_m += n_m[i]
        sum_d += d_p[i]

    a = sum_n_p / (1 + sum_d)
    b = sum_n_m / (1 + sum_d)

    for i in range(0, 5):
        bd_p[i] = d_p[i] * a
        bd_m[i] = d_m[i] * b
    # I don't know what just happened in this function. Maybe I will understand
    # later.



def transfer_columns(c1, c2, dest, bpp, c_index, height):
    """Transfers column pixel values to a pixel array.

    :param c1:
        The first column
    :param c2:
        The second column
    :param dest:
        The destination lightness array
    :param c_index:
        The index of the column in the target lightness array.
    :param bpp:
        Bytes per pixel
    :param height:
        The height of the column

    """
    s = [0]*bpp
    for row in range(0, height):
        for b in range(0, bpp):
            s[b] = c1[row][b] + c2[row][b]
            if s[b] > 255:
                s[b] = 255
            elif s[b] < 0:
                s[b] = 0

        hls = rgb_to_hls(s[0], s[1], s[2])
        dest[c_index][row] = hls[1]

def transfer_rows(r1, r2, dest, bpp, r_index, width):
    """Transfers row pixel values to a pixel array.

    :param r1:
        The first row
    :param r2:
        The second row
    :param dest:
        The destination lightness array
    :param c_index:
        The index of the column in the target lightness array.
    :param bpp:
        Bytes per pixel
    :param height:
        The height of the column

    """
    s = [0]*bpp
    for col in range(0, width):
        for b in range(0, bpp):
            s[b] = r1[col][b] + r2[col][b]
            if s[b] > 255:
                s[b] = 255
            elif s[b] < 0:
                s[b] = 0

        hls = rgb_to_hls(s[0], s[1], s[2])
        dest[col][r_index] = hls[1]


def cartoon_gimp(image):
    """Cartoonifies the given image

    :param image:
        The image to cartoonify, given as a PIL image.
    :returns:
        A new image that is cartoonified.

    """
    # Get width and height from the size tuple
    size = image.size
    width = size[0]
    height = size[1]

    # Bytes per pixel is found by taking the length of the first pixel.
    # Is this a hack?
    bpp = len(image.getbands())  
    has_alpha = bpp == 4

    # Copy the image and load the pixels for manipulation
    img = image.copy()
    pixels = img.load()

    # Prepare averaged and blurred image values.
    # These are called dest1 and dest2 in the GIMP code.
    avg = [[0]*height]*width
    blur = [[0]*height]*width

    # Calculate the standard deviations
    radius   = 1.0; # blur radius
    radius   = abs(radius) + 1.0;
    std_dev1 = sqrt(-(radius**2) / (2 * log(1.0 / 255.0)))
  
    radius   = cvals['mask_radius'];
    radius   = abs(radius) + 1.0;
    std_dev2 = sqrt(-(radius**2) / (2 * log(1.0 / 255.0)))

    # Derive the constants for calculating the gaussian from the std dev
    n_p1, n_m1 = [0]*5, [0]*5
    n_p2, n_m2 = [0]*5, [0]*5
    d_p1, d_m1 = [0]*5, [0]*5
    d_p2, d_m2 = [0]*5, [0]*5
    bd_p1, bd_m1 = [0]*5, [0]*5
    bd_p2, bd_m2 = [0]*5, [0]*5
    find_constants(n_p1, n_m1, d_p1, d_m1, bd_p1, bd_m1, std_dev1)
    find_constants(n_p2, n_m2, d_p2, d_m2, bd_p2, bd_m2, std_dev2)

    initial_p1 = [0]*bpp
    initial_p2 = [0]*bpp
    initial_m1 = [0]*bpp
    initial_m2 = [0]*bpp

    # First a vertical pass
    for col in range(0, width):
        # First and last pixels in the column
        p1 = pixels[col, 0]
        m1 = pixels[col, height-1]

        # Initialize the vectors that are going to hold the calculated pixel
        # values.
        vp1 = [[0,0,0]]*height
        vp2 = [[0,0,0]]*height
        vm1 = [[0,0,0]]*height
        vm2 = [[0,0,0]]*height

        # Set up the first values at the start and end of the column
        for b in range(0, bpp):
            initial_p1[b] = p1[b];
            initial_m1[b] = m1[b];

        # Iterate through all the rows
        for row in range(0, height):
            # Determine how many rows we can go back and forth from here.
            terms = row if row < 4 else 4

            # For each of the rows we manipulate the row pixel itself as well
            # as height-row pixel, ie. the pixels from the bottom of the column.
            reverse_row = height-1-row

            # Run through each band (r,g,b, [a])
            for b in range(0, bpp):
                j = 0
                for i in range(0, terms+1):
                    # Find the pixel color value from the front and the back.
                    px_row = pixels[col, row-i][b]
                    px_reverse_row = pixels[col, reverse_row+i][b]

                    # Set all the values
                    val = n_p1[i] * px_row - d_p1[i] * vp1[row-i][b]
                    vp1[row][b] += val

                    val = (n_m1[i] * px_reverse_row -
                           d_m1[i] * vm1[reverse_row+i][b])
                    vm1[row][b] += val

                    val = n_p2[i] * px_row - d_p2[i] * vp2[row-i][b]
                    vp2[row][b] += val

                    val = (n_m2[i] * px_reverse_row -
                           d_m2[i] * vm2[reverse_row+i][b])
                    vm2[row][b] += val
                    j = i

                for i in range(j, 5):
                    vp1[row][b] += (n_p1[i] - bd_p1[i]) * initial_p1[b]
                    vm1[row][b] += (n_m1[i] - bd_m1[i]) * initial_m1[b]
                    vp2[row][b] += (n_p2[i] - bd_p2[i]) * initial_p1[b]
                    vm2[row][b] += (n_m2[i] - bd_m2[i]) * initial_m1[b]
        # end row iteration
        transfer_columns(vp1, vm1, avg, bpp, col, height)
        transfer_columns(vp2, vm2, blur, bpp, col, height)
        if col % 20 == 0:
            logging.info('end of col iteration %s' % col)
    # end col iteration

    # Now a horizontal pass
    for row in range(0, height):
        # First and last pixels in the column
        p1 = pixels[0, row]
        m1 = pixels[width-1, row]

        # Initialize the vectors that are going to hold the calculated pixel
        # values.
        vp1 = [[0,0,0]]*width
        vp2 = [[0,0,0]]*width
        vm1 = [[0,0,0]]*width
        vm2 = [[0,0,0]]*width

        # Set up the first values at the start and end of the column
        for b in range(0, bpp):
            initial_p1[b] = p1[b];
            initial_m1[b] = m1[b];

        # Iterate through all the rows
        for col in range(0, width):
            # Determine how many columns we can go back and forth from here.
            terms = col if col < 4 else 4

            # For each of the rows we manipulate the row pixel itself as well
            # as height-row pixel, ie. the pixels from the bottom of the column.
            reverse_col = width-1-col

            # Run through each band (r,g,b, [a])
            for b in range(0, bpp):
                j = 0
                for i in range(0, terms+1):
                    # Find the pixel color value from the front and the back.
                    px_col = pixels[col-i, row][b]
                    px_reverse_col = pixels[reverse_col+i, row][b]

                    # Set all the values
                    val = n_p1[i] * px_col - d_p1[i] * vp1[col-i][b]
                    vp1[col][b] += val

                    val = (n_m1[i] * px_reverse_col -
                           d_m1[i] * vm1[reverse_col+i][b])
                    vm1[col][b] += val

                    val = n_p2[i] * px_col - d_p2[i] * vp2[col-i][b]
                    vp2[col][b] += val

                    val = (n_m2[i] * px_reverse_col -
                           d_m2[i] * vm2[reverse_col+i][b])
                    vm2[col][b] += val
                    j = i

                for i in range(j, 5):
                    vp1[col][b] += (n_p1[i] - bd_p1[i]) * initial_p1[b]
                    vm1[col][b] += (n_m1[i] - bd_m1[i]) * initial_m1[b]
                    vp2[col][b] += (n_p2[i] - bd_p2[i]) * initial_p1[b]
                    vm2[col][b] += (n_m2[i] - bd_m2[i]) * initial_m1[b]
        # end col iteration
        transfer_rows(vp1, vm1, avg, bpp, row, width)
        transfer_rows(vp2, vm2, blur, bpp, row, width)
        if row % 20 == 0:
            logging.info('end of row iteration %s' % row)
    # end row iteration

    ramp = compute_ramp(avg, blur, width, height, cvals['pct_black'])

    logging.info('ramp %s' % ramp)

    for col in range(0, width):
        for row in range(0, height):
            mult = 0.0
            if avg[col][row] != 0:
                diff = blur[col][row] / avg[col][row]
                if diff < cvals['threshold']:
                    if ramp == 0:
                        mult = 0.0
                    else:
                        mult = (ramp - min(ramp, cvals['threshold'] - diff)) / ramp
                else:
                    mult = 1.0

            lightness = max(0, min(blur[col][row]*mult, 255))

            if row*col % 100 == 0:
                pass
                #logging.info('lightness %s' % lightness)

            rgb = pixels[col, row]
            hls = rgb_to_hls(rgb[0], rgb[1], rgb[2])
            rgb = hls_to_rgb(hls[0], lightness, hls[2])
            pixels[col, row] = (int(rgb[0]), int(rgb[1]), int(rgb[2]))

    return img
