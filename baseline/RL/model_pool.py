from multiprocessing.shared_memory import SharedMemory, ShareableList
from multiprocessing import resource_tracker
import _pickle as cPickle
import time

def _unregister_shared_memory(name):
    # Python 3.8 tracks shared_memory in every attaching process.
    # Clients should detach without becoming the owner of unlink().
    try:
        resource_tracker.unregister('/' + name.lstrip('/'), 'shared_memory')
    except Exception:
        pass

class ModelPoolServer:
    
    def __init__(self, capacity, name):
        self.capacity = capacity
        self.n = 0
        self.model_list = [None] * capacity
        # shared_model_list: N metadata {id, _addr} + n
        metadata_size = 1024
        self.shared_model_list = ShareableList([' ' * metadata_size] * capacity + [self.n], name = name)
        self._closed = False
        
    def push(self, state_dict, metadata = {}):
        n = self.n % self.capacity
        if self.model_list[n]:
            # FIFO: release shared memory of older model
            old_memory = self.model_list[n].get('memory')
            if old_memory is not None:
                old_memory.close()
                try:
                    old_memory.unlink()
                except FileNotFoundError:
                    pass
        
        data = cPickle.dumps(state_dict) # model parameters serialized to bytes
        memory = SharedMemory(create = True, size = len(data))
        memory.buf[:] = data[:]
        # print('Created model', self.n, 'in shared memory', memory.name)
        
        metadata = metadata.copy()
        metadata['_addr'] = memory.name
        metadata['id'] = self.n
        self.model_list[n] = metadata
        self.shared_model_list[n] = cPickle.dumps(metadata)
        self.n += 1
        self.shared_model_list[-1] = self.n
        metadata['memory'] = memory

    def close(self):
        if self._closed:
            return
        for metadata in self.model_list:
            if not metadata:
                continue
            memory = metadata.get('memory')
            if memory is None:
                continue
            try:
                memory.close()
            except FileNotFoundError:
                pass
            try:
                memory.unlink()
            except FileNotFoundError:
                pass
        self.shared_model_list.shm.close()
        try:
            self.shared_model_list.shm.unlink()
        except FileNotFoundError:
            pass
        self._closed = True

class ModelPoolClient:
    
    def __init__(self, name):
        while True:
            try:
                self.shared_model_list = ShareableList(name = name)
                n = self.shared_model_list[-1]
                _unregister_shared_memory(self.shared_model_list.shm.name)
                break
            except:
                time.sleep(0.1)
        self.capacity = len(self.shared_model_list) - 1
        self.model_list = [None] * self.capacity
        self.n = 0
        self._closed = False
        self._update_model_list()
    
    def _update_model_list(self):
        n = self.shared_model_list[-1]
        if n > self.n:
            # new models available, update local list
            for i in range(max(self.n, n - self.capacity), n):
                self.model_list[i % self.capacity] = cPickle.loads(self.shared_model_list[i % self.capacity])
            self.n = n
    
    def get_model_list(self):
        self._update_model_list()
        model_list = []
        if self.n >= self.capacity:
            model_list.extend(self.model_list[self.n % self.capacity :])
        model_list.extend(self.model_list[: self.n % self.capacity])
        return model_list
    
    def get_latest_model(self):
        self._update_model_list()
        while self.n == 0:
            time.sleep(0.1)
            self._update_model_list()
        return self.model_list[(self.n + self.capacity - 1) % self.capacity]
        
    def load_model(self, metadata):
        self._update_model_list()
        n = metadata['id']
        if n < self.n - self.capacity: return None
        memory = SharedMemory(name = metadata['_addr'])
        try:
            state_dict = cPickle.loads(memory.buf)
        finally:
            memory.close()
            _unregister_shared_memory(memory.name)
        return state_dict

    def close(self):
        if self._closed:
            return
        self.shared_model_list.shm.close()
        self._closed = True
