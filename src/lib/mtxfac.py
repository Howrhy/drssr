import numpy
from random import randint
import time

from lib import stratus
from lib import util

def load_matrix_index(R):
    list_index = []
    for i in xrange(len(R)):
        for j in xrange(len(R[0])):
            if R[i][j] > 0:
                list_index.append(`i`+','+`j`)
    return list_index

def gd_update(Rij, Ui, Vj, alpha, lamb):
    e = numpy.dot(Ui.T,Vj) - Rij
    u_temp = Ui - alpha * ( (e * Vj) + (lamb * Ui) )
    v_temp = Vj - alpha * ( (e * Ui) + (lamb * Vj) )
    return u_temp, v_temp

def gd(R, U, V, steps=1800000, alpha=0.0001, lamb=0.002):

    list_index = load_matrix_index(R)

    percent = 0
    current_percent = 0

    start_time = time.time()

    for step in xrange(steps):
        
        for index in xrange(len(list_index)):

            sI,sJ =  list_index[index].split(',')

            i = int(sI)
            j = int(sJ)
        
            U[i], V[j] = gd_update(R[i][j], U[i,:], V[j,:], alpha, lamb)

        current_percent = util.calc_progress(steps, step+1, current_percent)

        # if(current_percent != percent):
        #     print current_percent
        #     percent = current_percent
        #     print (time.time() - start_time)/60
        #     start_time = time.time()

    return U, V

def sgd(R, U, V, steps=1800000, alpha=0.0001, lamb=0.002):

    list_index = load_matrix_index(R)

    len_list_index = len(list_index)

    for step in xrange(steps):

        index = randint(0,len_list_index-1)

        sI,sJ =  list_index[index].split(',')

        i = int(sI)
        j = int(sJ)

        U[i], V[j] = gd_update(R[i][j], U[i,:], V[j,:], alpha, lamb)

    return U, V

def dsgd(R, U, V, stratus_number, steps, alpha, lamb):

    T = 1000

    for step in xrange(T):

        list_stratus, list_U, list_V, index_pointer_c = stratus.split_matrix(R, U, V, stratus_number, step)

        for i in xrange(stratus_number):

            list_U[i],list_V[i] = sgd(list_stratus[i], list_U[i], list_V[i], steps, alpha, lamb)

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

    return U, V