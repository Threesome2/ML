# test.py

import numpy as np

def numpy_linspace():
    '''
    numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
    '''
    a = np.linspace(1,3,5)
    return a

def create_array():
    a = np.array([0,1,2,3,4])
    b = np.array((0,1,2,3,4))
    c = np.arange(5)
    d = np.linspace(0,2*np.pi,5)
    print(a)
    print(b)
    print(c)
    print(d)


def index_where():
    a = np.arange(0,100,10)
    b = np.where(a < 50)
    c = np.where(a >= 50)[0]
    print(b)
    print(c)

def string_test():
    string = ''
    string.find()


def main():
    index_where()





main()