from flask import Blueprint,jsonify,request
from models.param_RNA import Param_RNA
import json

model_bp = Blueprint("model",__name__,url_prefix="/model")
with open("process_cfg.json") as jsonf : 
    config = json.load(jsonf)
param_RNA = Param_RNA(config["model_name"])

@model_bp.route("/predict",methods = ["POST"])
def predict () :
    body = request.get_json()
    path = body.get("path",None)
    #leer path desde el body#
        
    results = param_RNA.test(path,sample_time=0.5)
    return jsonify(results)


    
    
