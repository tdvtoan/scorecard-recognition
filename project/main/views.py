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
            if excute not in VALID_CMD:
                return response_msg('ECOM')
            params = commands[1:]
            key = params[0]
            values = params[1:] if len(params) > 1 else []
            value = params[1] if len(params) > 1 else None
            if excute == 'set':
                ledis.string_set(key, value)
                result = 'OK'
            elif excute == 'get':
                result = ledis.string_get(key)
            elif excute == 'expire':
                pass
            elif excute == 'ttl':
                pass
            elif excute == 'del':
                ledis.del_key(key)
                result = 'OK'
            elif excute == 'flushdb':
                ledis.flush_db()
                result = 'OK'
            elif excute == 'llen':
                result = ledis.list_length(key)
            elif excute == 'rpush':
                ledis.list_rpush(key, values)
                result = 'OK'
            elif excute == 'lpop':
                result = ledis.list_lpop(key)
            elif excute == 'rpop':
                result = ledis.list_rpop(key)
            elif excute == 'lrange':
                result = ledis.list_lrange(key, values[0], values[1])
            elif excute == 'sadd':
                ledis.set_add(key, values)
                result = 'OK'
            elif excute == 'scard':
                result = ledis.set_scard(key)
            elif excute == 'smembers':
                result = ledis.set_smembers(key)
            elif excute == 'srem':
                ledis.set_srem(key, values)
                result = 'OK'
            elif excute == 'sinter':
                keys = [key]
                for val in values:
                    keys.append(val)
                ledis.set_sinter(keys)
                result = 'OK'
            elif excute == 'save':
                pass
            elif excute == 'restore':
                pass
        except:
            return jsonify(response)

    if result is None:
        return response_msg('EKTYP')
    return response_msg(result)



@main_blueprint.route("/about/")
def about():
    return render_template("main/about.html")
