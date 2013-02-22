"""
    handlers
    ========

    Contains most handlers for GIMPify.

    :copyright: 2013 by David Volquartz Lebech
    :license: MIT, see LICENSE for details.

"""
import os
import cStringIO
import logging

from PIL import Image, ImageFilter
import webapp2
import jinja2

from gimpy import cartoon

jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(os.path.abspath(
        os.path.join(os.path.dirname(__file__), 'templates'))))

class Home(webapp2.RequestHandler):
    def get(self):
        temp = jinja_env.get_template('index.html')
        self.response.write(temp.render())

    def post(self):
        # Read image.
        answer = self.request.POST['answer']
        if answer == '5' or answer == 'five':
            try:
                img_data = cStringIO.StringIO(self.request.POST['img_file'].value)
                img = Image.open(img_data)
            
                # Manipulate image
                img = cartoon.cartoon(img)
        
                # Save image to output stream.
                out_data = cStringIO.StringIO()
                img.save(out_data, 'JPEG')
                self.response.content_type = 'image/jpeg'
                self.response.write(out_data.getvalue())
            except Exception as e:
                logging.exception(e)
                self.response.write('It seems I cannot process your image')
        else:
            self.response.write('Wrong answer, type 5 or five.')


class CartoonExample(webapp2.RequestHandler):
    IMAGE = open('static/test.jpg').read()

    def get(self):
        # Read image.
        img_data = cStringIO.StringIO(self.IMAGE)
        img = Image.open(img_data)
    
        # Manipulate image
        img = cartoon.cartoon(img)

        # Save image to output stream.
        out_data = cStringIO.StringIO()
        img.save(out_data, 'JPEG')
        self.response.content_type = 'image/jpeg'
        self.response.write(out_data.getvalue())
