from flask import Flask,jsonify
from routes.model_routes import model_bp
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

app.register_blueprint(model_bp)
print(app.url_map)

if __name__ == "__main__" : 
    app.run(port=3001)