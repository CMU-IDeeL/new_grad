import numpy as np

'''
Do NOT change anything in the MemoryBuffer class. We are providing it to you. This class tracks
the gradients for each input in an operation.
'''
class MemoryBuffer():
    def __init__(self):
        self.memory = dict()

    @staticmethod
    def get_memory_loc(np_array):
        return np_array.__array_interface__['data'][0]

    def is_in_memory(self, np_array):
        return self.get_memory_loc(np_array) in self.memory

    def add_spot(self, np_array):
        if not self.is_in_memory(np_array):
            self.memory[self.get_memory_loc(np_array)] = np.zeros(np_array.shape)

    def update_param(self, np_array, gradient):
        # If a constant then no gradient is propagated (ex. mask in ReLU or labels in loss)
        if type(gradient).__name__=='NoneType':
            pass
        elif self.is_in_memory(np_array):
            self.memory[self.get_memory_loc(np_array)] += gradient
        else:
            raise Exception("Attempted to add gradient for a variable not in memory buffer.")

    def get_param(self, np_array):
        if self.is_in_memory(np_array):
            return self.memory[self.get_memory_loc(np_array)]
        else:
            raise Exception("Attempted to get gradient for a variable not in memory buffer.")

    def set_param(self, np_array, gradient):
        if self.is_in_memory(np_array):
            self.memory[self.get_memory_loc(np_array)] = gradient
        else:
            raise Exception("Attempted to set gradient for a variable not in memory buffer.")    

    def clear(self):
        self.memory = dict()