# project/main/views.py


#################
#### imports ####
#################
import urllib

import cv2
import flask
import numpy as np
from flask import render_template, Blueprint
from flask import Blueprint, jsonify, request
import json

################
#### config ####
################
from project import sc
from project.ml.ml import SCALE_IMG_WIDTH, SCALE_IMG_HEIGHT

main_blueprint = Blueprint('main', __name__,)


################
#### routes ####
################




@main_blueprint.route('/ml',methods=['POST'])
def home():
    data = json.loads(request.data)
    urls = data['image']
    response = {'result':[]}
    #i = 0
    for url in urls:
        resp = urllib.urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        image = cv2.resize(image, (650*2, SCALE_IMG_HEIGHT))
        par,memberId = sc.process(image)
        print('Par')
        print(par)
        response['result'].append({
            'par':par,
            'memberID':memberId
        })
        #i += 1
    return jsonify(response)


@main_blueprint.route("/about/")
def about():
    return render_template("main/about.html")
