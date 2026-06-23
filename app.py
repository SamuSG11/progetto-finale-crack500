from flask import Flask

from it.akron.home import home_bp

app = Flask(__name__)
app.register_blueprint(home_bp)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)