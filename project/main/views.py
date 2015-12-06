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
                pass
            elif excute == 'ttl':
                pass
            elif excute == 'del':
                ledis.del_key(key)
            elif excute == 'flushdb':
                ledis.flush_db()
            elif excute == 'llen':
                length = ledis.list_length(key)
                return jsonify({'response': length})
            elif excute == 'rpush':
                ledis.list_rpush(key, values)
            elif excute == 'lpop':
                result = ledis.list_lpop(key)
                return jsonify({'response': result})
            elif excute == 'rpop':
                result = ledis.list_rpop(key)
                return jsonify({'response': result})
            elif excute == 'lrange':
                result = ledis.list_lrange(key, values[0], values[1])
                return jsonify({'response': result})
            elif excute == 'sadd':
                ledis.set_add(key, values)
            elif excute == 'scard':
                result = ledis.set_scard(key)
                return jsonify({'response': result})
            elif excute == 'smembers':
                result = ledis.set_smembers(key)
                return jsonify({'response': result})
            elif excute == 'srem':
                ledis.set_srem(key, values)
            elif excute == 'sinter':
                keys = [key]
                for val in values:
                    keys.append(val)
                ledis.set_sinter(keys)
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
