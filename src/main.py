from lib import mtxfac
from lib import mtxfac_sr
from lib import dgd_sr
from lib import util

import numpy
import time
# import pylab

def exp_1():

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

    R = [
         [5,4,0,1,1,0,0,5],
         [0,5,3,0,1,1,0,4],
         [0,5,5,0,0,2,1,0],
         [0,1,1,3,5,4,0,1],
         [0,1,0,4,4,0,1,1],
         [1,1,0,5,5,0,0,0],
         [1,0,1,5,0,4,5,0]
        ]

    R = numpy.array(R)

    # users
    M = len(R)

    # itens
    N = len(R[0])

    K = 2

    U = numpy.random.rand(M,K)
    V = numpy.random.rand(N,K)

    U0 = numpy.copy(U);
    V0 = numpy.copy(V);

    U1 = numpy.copy(U);
    V1 = numpy.copy(V);

    alpha = 0.0002
    lamb  = 0.001
    steps = 1000
    
    print '###############  GD  ###############'

    start_time = time.time()

    nP0, nQ0, cost_f0 = mtxfac.gd(R, U0, V0, steps, alpha, lamb)
    nR0 = numpy.dot(nP0, nQ0.T)
    exp1 = util.rmse(nR0,R)

    time_exp1 = (time.time() - start_time)/60

    # print exp1

    # print time_exp1

    print '#############  GD SR  ############'

    start_time = time.time()

    nP1, nQ1 = mtxfac_sr.gd(R, U1, V1, steps, alpha, lamb, 0.001, 0, 0.001)
    # nP1, nQ1, cost_f1= mtxfac.sgd(R, U1, V1, steps * 25, alpha, lamb)
    nR1 = numpy.dot(nP1, nQ1.T)
    exp2 = util.rmse(nR1,R)

    time_exp2 = (time.time() - start_time)/60

    print exp1
    print exp2

    # pylab.plot(range(len(cost_f0)), cost_f0)
    # pylab.plot(range(len(cost_f1)), cost_f1)

    # pylab.show()
    
    # print time_exp2

def exp_2():

    R = numpy.loadtxt(open("../dataset/NY_MATRIX","rb"),delimiter=",")

    R = numpy.array(R)

    # users
    M = len(R)

    # itens
    N = len(R[0])

    K = 9

    U = numpy.loadtxt(open("../dataset/U","rb"),delimiter=",")
    V = numpy.loadtxt(open("../dataset/V","rb"),delimiter=",")

    U0 = numpy.copy(U);
    V0 = numpy.copy(V);

    U1 = numpy.copy(U);
    V1 = numpy.copy(V);

    U2 = numpy.copy(U);
    V2 = numpy.copy(V);

    U3 = numpy.copy(U); 
    V3 = numpy.copy(V);

    alpha = 0.0002
    lamb  = 0.001
    steps = 1000
    
    print '###############  GD  ###############'

    start_time = time.time()

    nP0, nQ0 = mtxfac.gd(R, U0, V0, steps, alpha, lamb)
    nR0 = numpy.dot(nP0, nQ0.T)
    exp1 = util.rmse(nR0,R)

    time_exp1 = (time.time() - start_time)/60

    print '##############  SGD  ##############'

    start_time = time.time()

    nP1, nQ1 = mtxfac.sgd(R, U1, V1, 15000000, alpha, lamb)
    nR1 = numpy.dot(nP1, nQ1.T)
    exp2 = util.rmse(nR1,R)

    time_exp2 = (time.time() - start_time)/60

    print '###########  DSGD  2x2  ###########'
    
    T = 1000
    steps_dsgd = 10000
    stratus_number = 2

    start_time = time.time()

    nP2, nQ2 = mtxfac.dsgd(R, U2, V2, stratus_number, T, steps_dsgd, alpha, lamb)
    nR2 = numpy.dot(nP2, nQ2.T)
    exp3 = util.rmse(nR2,R)

    time_exp3 = (time.time() - start_time)/60

    print '###########  DSGD  3x3  ###########'
    
    T = 1000
    steps_dsgd = 10000
    stratus_number = 3

    start_time = time.time()

    nP3, nQ3 = mtxfac.dsgd(R, U3, V3, stratus_number, T, steps_dsgd, alpha, lamb)
    nR3 = numpy.dot(nP3, nQ3.T)
    exp4 = util.rmse(nR3,R)

    time_exp4 = (time.time() - start_time)/60

    f_result = open('../dataset/result', 'w')
    f_result.write('gd   : '+`exp1`+'\n')
    f_result.write('time : '+`time_exp1`+'\n')
    f_result.write('sgd  : '+`exp2`+'\n')
    f_result.write('time : '+`time_exp2`+'\n')
    f_result.write('dsgd2x2 : '+`exp3`+'\n')
    f_result.write('time : '+`time_exp3`+'\n')
    f_result.write('dsgd3x3 : '+`exp4`+'\n')
    f_result.write('time : '+`time_exp4`+'\n')
    f_result.close()

def exp_3(R, U, V, steps):

    alpha = 0.0002
    lamb  = 0.001

    print '*******************  GD  *******************'

    U1 = numpy.copy(U);
    V1 = numpy.copy(V);

    start_time = time.time()

    nP1, nQ1, cost_f0 = mtxfac.gd(R, U1, V1, steps, alpha, lamb)
    nR1 = numpy.dot(nP1, nQ1.T)
    exp1 = util.rmse(nR1,R)

    time_exp1 = (time.time() - start_time)/60

    print '*******************  GD SR  *******************'

    U2 = numpy.copy(U);
    V2 = numpy.copy(V);
    
    start_time = time.time()

    nP2, nQ2 = dgd_sr.gd_default(R, U2, V2, steps, alpha, lamb, 0.001)
    nR2 = numpy.dot(nP2, nQ2.T)
    exp2 = util.rmse(nR2,R)

    time_exp2 = (time.time() - start_time)/60

    '******************* RESULT *******************'

    f_result = open('../dataset/result_gd_sr_'+`steps`, 'w')
    f_result.write('GD    : '+`exp1`+'\n')
    f_result.write('time  : '+`time_exp1`+'\n')
    f_result.write('GD SR : '+`exp2`+'\n')
    f_result.write('time  : '+`time_exp2`+'\n')
    f_result.close()

def exp_4():

    R = numpy.loadtxt(open("../dataset/NY_MATRIX","rb"),delimiter=",")

    R = numpy.array(R)

    # users
    M = len(R)

    # itens
    N = len(R[0])

    K = 9

    U = numpy.loadtxt(open("../dataset/U","rb"),delimiter=",")
    V = numpy.loadtxt(open("../dataset/V","rb"),delimiter=",")

    U1 = numpy.copy(U);
    V1 = numpy.copy(V);

    U2 = numpy.copy(U);
    V2 = numpy.copy(V);

    alpha = 0.0002
    lamb  = 0.001

    print '###########  DSGD 2x2  ###########'
    
    # T = 1000
    # steps_dsgd = 10000
    # stratus_number = 2

    # start_time = time.time()

    # nP1, nQ1 = mtxfac.dsgd(R, U1, V1, stratus_number, T, steps_dsgd, alpha, lamb)
    # nR1 = numpy.dot(nP1, nQ1.T)
    # exp1 = util.rmse(nR1,R)

    # time_exp1 = (time.time() - start_time)/60

    print '#########  DSGD 2x2 SR ##########'
    
    T = 1000
    steps_dsgd = 10000
    stratus_number = 2

    start_time = time.time()

    nP2, nQ2 = mtxfac_sr.dsgd(R, U2, V2, stratus_number, T, steps_dsgd, alpha, lamb)
    nR2 = numpy.dot(nP2, nQ2.T)
    exp2 = util.rmse(nR2,R)

    time_exp2 = (time.time() - start_time)/60

    f_result = open('../dataset/result_dsgd_sr', 'w')
    # f_result.write('dsgd2x2    : '+`exp1`+'\n')
    # f_result.write('time       : '+`time_exp1`+'\n')
    f_result.write('dsgd2x2 sr : '+`exp2`+'\n')
    f_result.write('time       : '+`time_exp2`+'\n')
    f_result.close()

def exp_5():

    R = [
         [5,4,0,1,1,0,0,5],
         [0,5,3,0,1,1,0,4],
         [0,5,5,0,0,2,1,0],
         [0,1,1,3,5,4,0,1],
         [0,1,0,4,4,0,1,1],
         [1,1,0,5,5,0,0,0],
         [1,0,1,5,0,4,5,0]
        ]

    R = numpy.array(R)

    # users
    M = len(R)

    # itens
    N = len(R[0])

    K = 2

    U = numpy.random.rand(M,K)
    V = numpy.random.rand(N,K)

    U0 = numpy.copy(U);
    V0 = numpy.copy(V);

    U1 = numpy.copy(U);
    V1 = numpy.copy(V);

    alpha = 0.0002
    lamb  = 0.001
    steps = 1000
    
    # print '###############  DGD  ###############'

    T = 1000
    steps_dsgd = 100
    stratus_number = 2
    beta = 0.001

    start_time = time.time()

    nP1, nQ1 = mtxfac.dgd(R, U1, V1, stratus_number, T, steps_dsgd, alpha, lamb)
    nR1 = numpy.dot(nP1, nQ1.T)
    exp1 = util.rmse(nR1,R)

    time_exp1 = (time.time() - start_time)/60

    # print '#############  DGD SR  ############'

    start_time = time.time()

    nP1, nQ1 = dgd_sr.dgd(R, U1, V1, stratus_number, T, steps_dsgd, alpha, lamb, beta)
    nR1 = numpy.dot(nP1, nQ1.T)
    exp2 = util.rmse(nR1,R)

    time_exp2 = (time.time() - start_time)/60

    print exp1
    print exp2

if __name__ == "__main__":

    R = numpy.loadtxt(open("../dataset/NY_MATRIX","rb"),delimiter=",")

    R = numpy.array(R)

    U = numpy.loadtxt(open("../dataset/U","rb"),delimiter=",")
    V = numpy.loadtxt(open("../dataset/V","rb"),delimiter=",")

    # exp_1()

    # exp_2()

    steps = 100
    for i in xrange(10):
        exp_3(R, U, V, steps)
        steps += 100

    # exp_4()

    # for i in xrange(10):
    #     print 'Exp - '+`i`
    #     exp_5()