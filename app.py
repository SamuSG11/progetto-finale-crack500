from flask import Flask

from it.akron.home import home_bp
from it.akron.api.dataset_bp import dataset_bp
from it.akron.api.augment_bp import augment_bp
from it.akron.api.optimize_bp import optimize_bp
from it.akron.api.verify_bp import verify_bp
from it.akron.api.training_bp import model_info_bp



app = Flask(__name__)
app.register_blueprint(home_bp)
app.register_blueprint(dataset_bp)
app.register_blueprint(augment_bp)
app.register_blueprint(optimize_bp)
app.register_blueprint(verify_bp)
app.register_blueprint(model_info_bp)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)