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

VALID_CMD = ['expire', 'ttl', 'del', 'flushdb','set','get','llen','rpush','lpop','rpop','lrange','sadd','scard','smembers','srem',
             'sinter','save','restore']
OK_RESPONSE = {'response':'OK'}
ECOM = {'reponse':'ECOM'}
@main_blueprint.route('/ledis',methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        response = {'response':'OK'}
        try:
            data = json.loads(request.data)
            if 'command' not in data:
                return jsonify(ECOM)
            commands = data['command'].split()
            excute = commands[0].lower()
            params = commands[1:]
            if excute == 'set':
                key = params[0]
                value = params[1]


            elif excute == 'get':
                value = params[0]

            elif excute == 'expire':
                key = params[0]
                seconds = params[1]

            elif excute == 'ttl':
                key = params[0]

            elif excute == 'del':
                key = params[0]

            elif excute == 'flushdb':
                pass

            elif excute == 'llen':
                key = params[0]

            elif excute == 'rpush':
                key = params[0]
                value = params[1]

            elif excute == 'lpop':
                key = params[0]

            elif excute == 'rpop':
                key = params[0]

            elif excute == 'lrange':
                key = params[0]
                start = params[1]
                stop = params[2]

            elif excute == 'sadd':
                key = params[0]
                value = params[1]

            elif excute == 'scard':
                key = params[0]

            elif excute == 'smembers':
                key = params[0]
                value = params[1]

            elif excute == 'srem':
                key = params

            elif excute == 'save':
                pass

            elif excute == 'restore':
                pass
        except:
            pass
        finally:
            return jsonify(response)
    elif request.method == 'GET':
        return jsonify(OK_RESPONSE)



@main_blueprint.route("/about/")
def about():
    return render_template("main/about.html")
