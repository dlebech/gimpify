"""
    gimpify
    =======

    GIMPing up those images.

    :copyright: 2013 by David Volquartz Lebech
    :license: MIT, see LICENSE for details.

"""
import webapp2
from webapp2_extras import routes

routes = [
        webapp2.Route('/',
            handler='handlers.Home',
            name='home'),
        routes.RedirectRoute('/cartoon',
            handler='handlers.CartoonExample',
            name='cartoon-example', strict_slash=True)
]

app = webapp2.WSGIApplication(routes, debug=False)
