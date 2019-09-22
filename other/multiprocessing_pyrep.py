from multiprocessing import Process, Queue
from pyrep import PyRep
from os.path import dirname, join, abspath
import os
from env_youbot import environment
import time

class test():
    def __init__(self):
        self.value = 5

    def __call__(self):
        print(self.value)
        print('Function class executed')
        return self.value

def launch_pyrep(name,t):
        a = t()
        print(a+1)

if __name__ == '__main__':
    t = test()
    print(t())
    p = Process(target=launch_pyrep,args=('valid',t))
    p.start()
    p.join()

