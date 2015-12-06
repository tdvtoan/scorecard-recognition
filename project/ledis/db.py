from blist import sortedset
import time
import simplejson
import json
import os

class LedisDb(object):

    def __init__(self, storage_folder=None, option=None):
        self.storage_folder = storage_folder
        self.option = option
        self._data = {}
        self._expired = {}

    def is_exists(self, key):
        return key in self._data

    def is_string(self, key):
         return isinstance(self._data[key], basestring)

    def is_list(self, key):
        return isinstance(self._data[key], list)

    def is_set(self, key):
        return isinstance(self._data[key], sortedset)

    def restore(self):
        pass

    def del_key(self, key):
        if self.is_exists(key):
            del self._data[key]

    def flush_db(self):
        print 'Flushed'
        self._data.clear()

    def string_set(self, key, value):
        if self.is_exists(key) and not self.is_string(key):
            return None
        self._data[key] = value

    def string_get(self, key):
        return self._data[key] if self.is_exists(key) and self.is_string(key) else None

    def list_length(self, key):
        if self.is_exists(key) and self.is_list(key):
            return len(self._data[key])

    def list_rpush(self, key, values):
        if not self.is_exists(key):
            self._data[key] = []
        for val in values:
            self._data[key].append(val)
        return len(self._data[key])

    def list_lpop(self, key):
        if self.is_exists(key) and self.is_list(key) and len(self._data[key]) > 0:
            val = self._data[key][0]
            del self._data[key][0]
            return val
        return None

    def list_rpop(self, key):
        if self.is_exists(key) and self.is_list(key) and len(self._data[key]) > 0:
            val = self._data[key][-1]
            del self._data[key][-1]
            return val
        return None

    def list_lrange(self, key, start, stop):
        if self.is_exists(key) and self.is_list(key):
            l = self._data[key]
            return l[int(start):int(stop)+1]

    def set_add(self, key, values):
        if not self.is_exists(key):
            self._data[key] = sortedset([])
        elif self.is_set(key):
            return None
        for val in values:
            self._data[key].add(val)
        return True

    def set_scard(self, key):
        if self.is_exists(key) and self.is_set(key):
            return len(self._data[key])
        return None

    def set_smembers(self, key):
        if self.is_exists(key) and self.is_set(key):
            return [item for item in self._data[key]]
        return None

    def set_srem(self, key, values):
        if self.is_exists(key) and self.is_set(key):
            for val in values:
                self._data[key].discard(val)
        return None

    def set_sinter(self, keys):
        result = sortedset()
        for key in keys:
            if self.is_exists(key) and self.is_list(key):
                result.intersection(self._data[key])
        return [item for item in result]

    def expire_key(self, key, seconds):
        self._expired[key] = int(time.time()) + int(abs(seconds))

    def clear_expired(self):
        now = int(time.time())
        for key in self._expired:
            if self._expired[key] < now and self.is_exists(key):
                del self._data[key]

    def restore(self):
        file_name = os.environ.get('LEDIS_PATH', None) + '/latest.json'
        with open(file_name) as data_file:
            self._data = json.load(data_file)


    def save(self):
        now = int(time.time())
        data_types = {}
        for key in self._data:
            type = 'string'
            if self.is_list(key):
                type = 'list'
            elif self.is_set(key):
                type = 'set'
            data_types[key] = type
        structure_file_name = os.environ.get('LEDIS_PATH', None) + '/latest.dson'
        file_name = os.environ.get('LEDIS_PATH', None) + '/latest.json'
        simplejson.dump(self._data, open(file_name, 'wb'))
        simplejson.dump(data_types, open(structure_file_name, 'wb'))
        return True
