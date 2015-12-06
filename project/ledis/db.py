from blist import sortedset


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
            return l[int(start):int(stop)]

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