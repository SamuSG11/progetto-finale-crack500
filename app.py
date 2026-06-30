from flask import Flask

from it.akron.api.temp.augment_bp import augment_bp
from it.akron.api.temp.optimize_bp import optimize_bp
from it.akron.api.temp.verify_bp import verify_bp

from it.akron.api.outputs.predict_bp import predict_bp
from it.akron.api.outputs.info_bp import dataset_bp 
from it.akron.api.outputs.model_info_bp import model_info_bp
from it.akron.api.temp.home import home_bp
from it.akron.api.temp.notebook_bp import notebook_bp
from it.akron.api.temp.training_bp import training_bp




app = Flask(__name__)
app.register_blueprint(dataset_bp)
app.register_blueprint(augment_bp)
app.register_blueprint(optimize_bp)
app.register_blueprint(verify_bp)
app.register_blueprint(model_info_bp)
app.register_blueprint(home_bp)
app.register_blueprint(predict_bp)
app.register_blueprint(notebook_bp)
app.register_blueprint(training_bp)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)