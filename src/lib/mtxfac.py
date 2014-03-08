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

        # print 'DEBUG: '+`i`+','+`j` 

        U[i], V[j] = gd_update(R[i][j], U[i,:], V[j,:], alpha, lamb)

    return U, V

def dsgd_old(R, U, V, stratus_number, steps, alpha, lamb):

    T = 100

    for step in xrange(T):

        list_stratus, list_U, list_V, index_stratus_selected = stratus.split_matrix_old(R, U, V, stratus_number, step)

        # print 'list_stratus'
        # print list_stratus

        # print 'list_U'
        # print list_U

        # print 'list_V'
        # print list_V

        for i in xrange(stratus_number):

            list_U[i],list_V[i] = gd(list_stratus[i], list_U[i], list_V[i], steps, alpha, lamb)

        # print 'list_V_MODIFIED'
        # print list_V

        index_U=0
        for index_array in xrange(stratus_number):
            temp_U = list_U[index_array]
            
            for i in xrange(len(temp_U)):
                for j in xrange(len(temp_U[0])):
                    U[index_U][j] = temp_U[i][j]
                index_U += 1

        index_V=0
        for x in xrange(stratus_number):
            temp_V = list_V[x]
            index_array = index_stratus_selected[x] * len(temp_V)

            for i in xrange(len(temp_V)):
                for j in xrange(len(temp_V[0])):
                    V[index_array + i][j] = temp_V[i][j]

    # print 'FINAL V'
    # print V

    return U, V

def dsgd(R, U, V, stratus_number, steps, alpha, lamb):

    T = 1000

    for step in xrange(T):

        list_stratus, list_U, list_V, index_pointer_c = stratus.split_matrix(R, U, V, stratus_number, step)

        # print 'list_stratus'
        # print list_stratus

        # print 'list_U'
        # print list_U

        # print 'list_V'
        # print list_V

        # print 'index_pointer_c'
        # print index_pointer_c

        for i in xrange(stratus_number):

            list_U[i],list_V[i] = gd(list_stratus[i], list_U[i], list_V[i], steps, alpha, lamb)

        # print 'list_V_MODIFIED'
        # print list_V

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
                # print temp_V[i]
                V[index_V+i] = temp_V[i]

    # print 'FINAL V'
    # print V

    return U, V

def dsgd_3x3(R, U, V, alpha, lamb):

    v_split_R = numpy.split(R,3)
    
    h_split_0 = numpy.hsplit(v_split_R[0],3)
    h_split_1 = numpy.hsplit(v_split_R[1],3)
    h_split_2 = numpy.hsplit(v_split_R[2],3)

    b_11 = h_split_0[0]
    b_12 = h_split_0[1]
    b_13 = h_split_0[2]

    b_21 = h_split_1[0]
    b_22 = h_split_1[1]
    b_23 = h_split_1[2]

    b_31 = h_split_2[0]
    b_32 = h_split_2[1]
    b_33 = h_split_2[2]

    split_U = numpy.split(U,3)
    split_V = numpy.split(V,3)

    T = 100
    
    U1 = split_U[0]
    U2 = split_U[1]
    U3 = split_U[2]

    V1 = split_V[0]
    V2 = split_V[1]
    V3 = split_V[2]

    i = 0

    for step in xrange(1000): 

        i = randint(0,5)

        if i == 0:

            U1,V1 = sgd(b_11, U1, V1, T, alpha, lamb)
            U2,V2 = sgd(b_22, U2, V2, T, alpha, lamb)
            U3,V3 = sgd(b_33, U3, V3, T, alpha, lamb)

        if i == 1:

            U1,V2 = sgd(b_12, U1, V2, T, alpha, lamb)
            U2,V3 = sgd(b_23, U2, V3, T, alpha, lamb)
            U3,V1 = sgd(b_31, U3, V1, T, alpha, lamb)

        if i == 2:
            U1,V3 = sgd(b_13, U1, V3, T, alpha, lamb)
            U2,V1 = sgd(b_21, U2, V1, T, alpha, lamb)
            U3,V2 = sgd(b_32, U3, V2, T, alpha, lamb)

        if i == 3:
            U1,V1 = sgd(b_11, U1, V1, T, alpha, lamb)
            U2,V3 = sgd(b_23, U2, V3, T, alpha, lamb)
            U3,V2 = sgd(b_32, U3, V2, T, alpha, lamb)

        if i == 4:
            U1,V2 = sgd(b_12, U1, V2, T, alpha, lamb)
            U2,V1 = sgd(b_21, U2, V1, T, alpha, lamb)
            U3,V3 = sgd(b_33, U3, V3, T, alpha, lamb)

        if i == 5:
            U1,V3 = sgd(b_13, U1, V3, T, alpha, lamb)
            U2,V2 = sgd(b_22, U2, V2, T, alpha, lamb)
            U3,V1 = sgd(b_31, U3, V1, T, alpha, lamb)

    U = numpy.concatenate((U1, U2, U3))
    V = numpy.concatenate((V1, V2, V3))

    return U, V

def dgd_3x3(R, U, V, alpha, lamb):

    v_split_R = numpy.split(R,3)
    
    h_split_0 = numpy.hsplit(v_split_R[0],3)
    h_split_1 = numpy.hsplit(v_split_R[1],3)
    h_split_2 = numpy.hsplit(v_split_R[2],3)

    b_11 = h_split_0[0]
    b_12 = h_split_0[1]
    b_13 = h_split_0[2]

    b_21 = h_split_1[0]
    b_22 = h_split_1[1]
    b_23 = h_split_1[2]

    b_31 = h_split_2[0]
    b_32 = h_split_2[1]
    b_33 = h_split_2[2]

    split_U = numpy.split(U,3)
    split_V = numpy.split(V,3)

    T = 100
    
    U1 = split_U[0]
    U2 = split_U[1]
    U3 = split_U[2]

    V1 = split_V[0]
    V2 = split_V[1]
    V3 = split_V[2]

    i = 0

    for step in xrange(1000): 

        i = randint(0,5)

        if i == 0:

            U1,V1 = gd(b_11, U1, V1, T, alpha, lamb)
            U2,V2 = gd(b_22, U2, V2, T, alpha, lamb)
            U3,V3 = gd(b_33, U3, V3, T, alpha, lamb)

        if i == 1:

            U1,V2 = gd(b_12, U1, V2, T, alpha, lamb)
            U2,V3 = gd(b_23, U2, V3, T, alpha, lamb)
            U3,V1 = gd(b_31, U3, V1, T, alpha, lamb)

        if i == 2:
            U1,V3 = gd(b_13, U1, V3, T, alpha, lamb)
            U2,V1 = gd(b_21, U2, V1, T, alpha, lamb)
            U3,V2 = gd(b_32, U3, V2, T, alpha, lamb)

        if i == 3:
            U1,V1 = gd(b_11, U1, V1, T, alpha, lamb)
            U2,V3 = gd(b_23, U2, V3, T, alpha, lamb)
            U3,V2 = gd(b_32, U3, V2, T, alpha, lamb)

        if i == 4:
            U1,V2 = gd(b_12, U1, V2, T, alpha, lamb)
            U2,V1 = gd(b_21, U2, V1, T, alpha, lamb)
            U3,V3 = gd(b_33, U3, V3, T, alpha, lamb)

        if i == 5:
            U1,V3 = gd(b_13, U1, V3, T, alpha, lamb)
            U2,V2 = gd(b_22, U2, V2, T, alpha, lamb)
            U3,V1 = gd(b_31, U3, V1, T, alpha, lamb)

    U = numpy.concatenate((U1, U2, U3))
    V = numpy.concatenate((V1, V2, V3))

    return U, V