from lib import mtxfac
from lib import mtxfac_sr
from lib import util

import numpy
import time

if __name__ == "__main__":

    # matrix m x n
    # R = [
    #      [5,4,0,1,3,0],
    #      [5,0,4,0,0,1],
    #      [0,5,0,1,1,0],
    #      [1,0,1,5,4,0],
    #      [0,1,0,0,5,4],
    #      [1,0,2,5,0,0]
    #     ]

    # R = [
    #      [5,4,0,1,3],
    #      [5,0,4,0,0],
    #      [0,5,0,1,1],
    #      [1,0,1,5,4],
    #      [0,1,0,0,5],
    #      [1,0,2,5,0]
    #     ]

    # R = [
    #      [5,4,0,1,3,0],
    #      [5,0,4,0,0,0],
    #      [0,5,0,1,1,0],
    #      [1,0,1,5,4,0],
    #      [0,1,0,0,5,0]
    #     ]

    # R = [
    #      [5,4,0,1,1,0,0,5],
    #      [0,5,3,0,1,1,0,4],
    #      [0,5,5,0,0,2,1,0],
    #      [0,1,1,3,5,4,0,1],
    #      [0,1,0,4,4,0,1,1],
    #      [1,1,0,5,5,0,0,0],
    #      [1,0,1,5,0,4,5,0]
    #     ]    

    R = numpy.loadtxt(open("../dataset/NY_MATRIX","rb"),delimiter=",")

    R = numpy.array(R)

    # users
    M = len(R)

    # itens
    N = len(R[0])

    K = 9

    # util.generate_U_V(M,N,K)

    U = numpy.loadtxt(open("../dataset/U","rb"),delimiter=",")
    V = numpy.loadtxt(open("../dataset/V","rb"),delimiter=",")

    U0 = numpy.copy(U);
    V0 = numpy.copy(V);

    U1 = numpy.copy(U); 
    V1 = numpy.copy(V);

    U2 = numpy.copy(U); 
    V2 = numpy.copy(V);

    alpha = 0.0002
    lamb  = 0.001
    steps = 10000
    stratus_number = 2
    
    start_time = time.time()

    nP0, nQ0 = mtxfac.gd(R, U0, V0, steps, alpha, lamb)
    nR0 = numpy.dot(nP0, nQ0.T)
    exp1 = util.rmse(nR0,R)

    time_exp1 = (time.time() - start_time)/60

    start_time = time.time()

    nP1, nQ1 = mtxfac.sgd(R, U1, V1, 200000, alpha, lamb)
    nR1 = numpy.dot(nP1, nQ1.T)
    exp2 = util.rmse(nR1,R)

    time_exp2 = (time.time() - start_time)/60

    start_time = time.time()

    nP2, nQ2 = mtxfac.dsgd(R, U2, V2, stratus_number, 100, alpha, lamb)
    nR2 = numpy.dot(nP2, nQ2.T)
    exp3 = util.rmse(nR2,R)

    time_exp3 = (time.time() - start_time)/60

    f_result = open('../dataset/result', 'w')
    f_result.write('gd   : '+`exp1`+'\n')
    f_result.write('time : '+`time_exp1`+'\n')
    f_result.write('sgd  : '+`exp2`+'\n')
    f_result.write('time : '+`time_exp2`+'\n')
    f_result.write('dsgd : '+`exp3`+'\n')
    f_result.write('time : '+`time_exp3`+'\n')
    f_result.close()

    # nP1, nQ1 = mtxfac_sr.gd_sr(R, U1, V1, steps, alpha, lamb, 0.001)
    # nR1 = numpy.dot(nP1, nQ1.T)
    # print util.rmse(nR1,R)