"""
    handlers
    ========

    Contains most handlers for GIMPify.

    :copyright: 2013-2018 by David Volquartz Lebech
    :license: MIT, see LICENSE for details.

"""
import io
import logging

from PIL import Image, ImageFilter
from flask import render_template, request

from gimpy import cartoon


logger = logging.getLogger(__name__)


def home():
    return render_template('index.html')


def home_upload_image():
    try:
        f = request.files['img_file']
        img = Image.open(f)

        logger.info('Width {}, height {}'.format(img.width, img.height))
        if img.width > 800 or img.height > 800:
            logger.info('Resizing image')
            img.thumbnail((800, 800)) # thumbnail changes the image in place

        # Manipulate image
        img = cartoon.cartoon(img)

        # Save image to output stream.
        out_data = io.BytesIO()
        img.save(out_data, 'JPEG')
        return out_data.getvalue(), { 'Content-type': 'image/jpeg' }
    except Exception as e:
        logger.exception(e)
        return 'It seems I cannot process your image'


def example_image():
    img = Image.open('static/img/test.jpg')

    # Manipulate image
    img = cartoon.cartoon(img)

    # Save image to output stream.
    out_data = io.BytesIO()
    img.save(out_data, 'JPEG')
    return out_data.getvalue(), { 'Content-type': 'image/jpeg' }