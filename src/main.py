from lib import mtxfac
from lib import mtxfac_sr
from lib import util
import numpy

if __name__ == "__main__":

    # matrix m x n
    R = [
         [5,4,0,1,3,0],
         [5,0,4,0,0,1],
         [0,5,0,1,1,0],
         [1,0,1,5,4,0],
         [0,1,0,0,5,4],
         [1,0,2,5,0,0]
        ]

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

    # R = numpy.loadtxt(open("../dataset/NY_MATRIX","rb"),delimiter=",")

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

    U2 = numpy.copy(U); 
    V2 = numpy.copy(V);


    alpha = 0.0002
    lamb  = 0.001
    steps = 18000
    stratus_number = 3

    # print V

    nP0, nQ0 = mtxfac.gd(R, U0, V0, steps, alpha, lamb)
    nR0 = numpy.dot(nP0, nQ0.T)
    print util.rmse(nR0,R)

    # nP1, nQ1 = mtxfac.sgd(R, U1, V1, steps, alpha, lamb)
    # nR1 = numpy.dot(nP1, nQ1.T)
    # print util.rmse(nR1,R)

    # nP1, nQ1 = mtxfac.dsgd_old(R, U1, V1, stratus_number, 10000, alpha, lamb)
    # nR1 = numpy.dot(nP1, nQ1.T)
    # print util.rmse(nR1,R)

    nP1, nQ1 = mtxfac.dgd_3x3(R, U1, V1, alpha, lamb)
    nR1 = numpy.dot(nP1, nQ1.T)
    print util.rmse(nR1,R)

    nP2, nQ2 = mtxfac.dsgd(R, U2, V2, stratus_number, 100, alpha, lamb)
    nR2 = numpy.dot(nP2, nQ2.T)
    print util.rmse(nR2,R)

    # nP1, nQ1 = mtxfac_sr.gd_sr(R, U1, V1, steps, alpha, lamb, 0.01)
    # nR1 = numpy.dot(nP1, nQ1.T)
    # print util.rmse(nR1,R)