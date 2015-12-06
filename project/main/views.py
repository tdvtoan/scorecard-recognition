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
from project import ledis

main_blueprint = Blueprint('main', __name__,)


################
#### routes ####
################

VALID_CMD = ['expire', 'ttl', 'del', 'flushdb','set','get','llen','rpush','lpop','rpop','lrange','sadd','scard','smembers','srem',
             'sinter','save','restore']
ECOM = {'reponse':'ECOM'}

def response_msg(msg):
    return jsonify({'response': msg})

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
            key = params[0]
            values = params[1:] if len(params) > 1 else []
            value = params[1] if len(params) > 1 else None
            if excute == 'set':
                ledis.string_set(key, value)
            elif excute == 'get':
                val = ledis.string_get(key)
                return jsonify({'response': val})
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
                key = params[0]


            elif excute == 'save':
                pass

            elif excute == 'restore':
                pass
        except:
            return jsonify(response)
    return response_msg('OK')



@main_blueprint.route("/about/")
def about():
    return render_template("main/about.html")
