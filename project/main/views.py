# project/main/views.py


#################
#### imports ####
#################

from flask import render_template, Blueprint
from flask import Blueprint, jsonify, request
import json
################
#### config ####
################

main_blueprint = Blueprint('main', __name__,)


################
#### routes ####
################


@main_blueprint.route('/ledis',methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        data = {'response':'OK'}
        try:
            data = json.loads(request.data)
        except:
            pass
        finally:
            return jsonify(data)
    elif request.method == 'GET':
        return jsonify({'response':'OK'})



@main_blueprint.route("/about/")
def about():
    return render_template("main/about.html")
