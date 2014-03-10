import numpy
from random import randint

from lib import mtxfac
from lib import util
from lib import stratus

def load_grafo_social(R):
    grafo_size = len(R)

    social_graph = numpy.zeros((grafo_size, grafo_size))

    # social_network = numpy.loadtxt(open("../dataset/SN_TEST","rb"),delimiter=",")
    social_network = numpy.loadtxt(open("../dataset/NY_SN","rb"),delimiter=",")

    for i in xrange(len(social_network)):
        
        user   = social_network[i][0]
        friend = social_network[i][1]

        x = R[user]
        y = R[friend]

        cor_pearson = util.pearson(x,y)

        social_graph[user][friend] = cor_pearson
        social_graph[friend][user] = cor_pearson

    # print social_graph

    return social_graph

def sr_f(i, P, SG):
    reg = 0

    for f in xrange(len(SG[i])):
        if SG[i][f] > 0:
            reg += SG[i][f] * (P[i] - P[f])

    return reg

def gd_default(R, U, V, steps, alpha, lamb, beta):

    percent = 0
    current_percent = 0

    SG = load_grafo_social(R)

    list_index = mtxfac.load_matrix_index(R)

    len_list_index = len(list_index)

    for step in xrange(steps):
        
        for index in xrange(len(list_index)):

            sI,sJ =  list_index[index].split(',')

            i = int(sI)
            j = int(sJ)

            e = numpy.dot(U[i].T,V[j]) - R[i][j]
            u_temp = U[i] - alpha * ( (e * V[j]) + (lamb * U[i]) + beta*sr_f(i,U,SG) )
            V[j]   = V[j] - alpha * ( (e * U[i]) + (lamb * V[j]) )
            U[i]   = u_temp

        current_percent = util.calc_progress(steps, step+1, current_percent)

        if(current_percent != percent):
            print current_percent
            percent = current_percent
        
    return U, V

def gd(R, U, V, steps, alpha, lamb, beta, SG, FULL_U, REAL_INDEX):

    percent = 0
    current_percent = 0

    list_index = mtxfac.load_matrix_index(R)

    len_list_index = len(list_index)

    for step in xrange(steps):
        
        for index in xrange(len(list_index)):

            sI,sJ =  list_index[index].split(',')

            i = int(sI)
            j = int(sJ)

            e = numpy.dot(U[i].T,V[j]) - R[i][j]
            u_temp = U[i] - alpha * ( (e * V[j]) + (lamb * U[i]) + beta*sr_f(REAL_INDEX + i, FULL_U, SG) )
            V[j]   = V[j] - alpha * ( (e * U[i]) + (lamb * V[j]) )
            U[i]   = u_temp

        # current_percent = util.calc_progress(steps, step+1, current_percent)

        # if(current_percent != percent):
        #     print current_percent
        #     percent = current_percent
        
    return U, V    


def dgd(R, U, V, stratus_number, T, steps, alpha, lamb, beta):

    percent = 0
    current_percent = 0

    SG = load_grafo_social(R)

    for step in xrange(T):

        list_stratus, list_U, list_V, index_pointer_r, index_pointer_c = stratus.split_matrix(R, U, V, stratus_number, step)

        for i in xrange(stratus_number):

            list_U[i],list_V[i] = gd(list_stratus[i], list_U[i], list_V[i], steps, alpha, lamb, beta, SG, U, index_pointer_r[i])

        index_U=0
        for index_array in xrange(stratus_number):
            temp_U = list_U[index_array]
            
            for i in xrange(len(temp_U)):
                for j in xrange(len(temp_U[0])):
                    U[index_U][j] = temp_U[i][j]
                index_U += 1

        index_V=0
        for x in xrange(stratus_number):
            index_V = index_pointer_c[x]

            temp_V = list_V[x]

            for i in xrange(len(temp_V)):
                V[index_V+i] = temp_V[i]

        current_percent = util.calc_progress(T, step+1, current_percent)

        # if(current_percent != percent):
        #     print current_percent
        #     percent = current_percent

    return U, V