import numpy
from random import randint

from lib import mtxfac
from lib import util

def load_grafo_social(R):
    grafo_size = len(R)

    social_graph = numpy.zeros((grafo_size, grafo_size))

    social_network = numpy.loadtxt(open("../dataset/SN_TEST","rb"),delimiter=",")

    for i in xrange(len(social_network)):
        
        user   = social_network[i][0]
        friend = social_network[i][1]

        x = R[user]
        y = R[friend]

        # print 'User  : '+`user`+','+`x` 
        # print 'Friend: '+`friend`+','+`y` 

        cor_pearson = util.pearson(x,y)

        social_graph[user][friend] = cor_pearson
        social_graph[friend][user] = cor_pearson

    return social_graph

def sr_f(i, P, SG):
    reg = 0.0

    for f in xrange(len(SG[i])):
        if SG[i][f] > 0:
            reg += SG[i][f] * (P[i] - P[f])

    return reg

def gd_update(Rij, U, index_U, Vj, SG, alpha, lamb, beta):
    Ui = U[index_U,:]
    e = numpy.dot(Ui.T,Vj) - Rij
    u_temp = Ui - alpha * ( (e * Vj) + (lamb * Ui) + beta*sr_f(index_U,U,SG) )
    v_temp = Vj - alpha * ( (e * Ui) + (lamb * Vj) )
    return u_temp, v_temp

def gd_sr(R, U, V, steps=1800000, alpha=0.0001, lamb=0.002, beta=0.001):

    list_index = mtxfac.load_matrix_index(R)

    len_list_index = len(list_index)

    SG = load_grafo_social(R)

    for step in xrange(steps):
        
        for index in xrange(len(list_index)):

            sI,sJ =  list_index[index].split(',')

            i = int(sI)
            j = int(sJ)

            U[i], V[j] = gd_update(R[i][j], U, i, V[j,:], SG, alpha, lamb, beta)
        
    return U, V    

def sgd_sr(R, U, V, steps=1800000, alpha=0.0001, lamb=0.002, beta=0.001):

    list_index = mtxfac.load_matrix_index(R)

    len_list_index = len(list_index)

    SG = load_grafo_social(R)

    for step in xrange(steps):

        index = randint(0,len_list_index-1)

        sI,sJ =  list_index[index].split(',')

        i = int(sI)
        j = int(sJ)

        U[i], V[j] = gd_update(R[i][j], U, i, V[j,:], alpha, lamb, beta)

    return U, V