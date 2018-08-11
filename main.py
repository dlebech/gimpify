"""
    gimpify
    =======

    GIMPing up those images.

    :copyright: 2013-2018 by David Volquartz Lebech
    :license: MIT, see LICENSE for details.

"""
import logging

from flask import Flask

import handlers


app = Flask(__name__)

app.route('/', methods=['GET'])(handlers.home)
app.route('/', methods=['POST'])(handlers.home_upload_image)
app.route('/example', methods=['GET'])(handlers.example_image)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    app.run(port=8080, debug=True)