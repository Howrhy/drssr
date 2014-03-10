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

    cost = e ** 2
    # print 'variance: '+`abs(u_temp-Ui)`

    return u_temp, v_temp, cost

def gd(R, U, V, steps, alpha, lamb):

    percent = 0
    current_percent = 0

    cost_f = []

    # start_time = time.time()

    list_index = load_matrix_index(R)

    for step in xrange(steps):
        
        cost_sum = 0

        for index in xrange(len(list_index)):

            sI,sJ =  list_index[index].split(',')

            i = int(sI)
            j = int(sJ)
        
            U[i], V[j], cost = gd_update(R[i][j], U[i,:], V[j,:], alpha, lamb)

            cost_sum += cost

        cost_f.append(cost_sum)

        current_percent = util.calc_progress(steps, step+1, current_percent)

        if(current_percent != percent):
            print current_percent
            percent = current_percent
        #     print (time.time() - start_time)/60
        #     start_time = time.time()

    return U, V, cost_f

def sgd(R, U, V, steps=1800000, alpha=0.0001, lamb=0.002):

    # percent = 0
    # current_percent = 0

    cost_f = []

    list_index = load_matrix_index(R)

    len_list_index = len(list_index)

    cost_sum = 0

    for step in xrange(steps):

        index = randint(0,len_list_index-1)

        sI,sJ =  list_index[index].split(',')

        i = int(sI)
        j = int(sJ)

        U[i], V[j], cost = gd_update(R[i][j], U[i,:], V[j,:], alpha, lamb)

        cost_sum = 0

        if step % 25 == 0:

            for index1 in xrange(len(list_index)):

                sI1,sJ1 =  list_index[index1].split(',')

                i1 = int(sI1)
                j1 = int(sJ1)

                e = numpy.dot(U[i1,:], V[j1,:]) - R[i1][j1]

                cost_sum += e ** 2

            cost_f.append(cost_sum)

        # current_percent = util.calc_progress(steps, step+1, current_percent)

        # if(current_percent != percent):
        #     print current_percent
        #     percent = current_percent

    return U, V, cost_f

def sgd1(R, U, V, steps=1800000, alpha=0.0001, lamb=0.002):

    cost_f = []

    list_index = load_matrix_index(R)

    z = extract_train(list_index)

    loop = 0

    for step in xrange(steps):

        z = extract_train(list_index)

        cost_sum = 0

        for index in xrange(len(z)):

            loop += 1

            sI,sJ =  z[index].split(',')

            i = int(sI)
            j = int(sJ)
        
            U[i], V[j], cost = gd_update(R[i][j], U[i,:], V[j,:], alpha, lamb)

            cost_sum += cost

        cost_f.append(cost_sum)

    print loop

    return U, V, cost_f

def sgd2(R, U, V, steps=1800000, alpha=0.0001, lamb=0.002):


    cost_f = []

    list_index = load_matrix_index(R)

    z = numpy.random.permutation(list_index)

    for step in xrange(steps):
        
        cost_sum = 0

        for index in xrange(len(z)):

            sI,sJ =  z[index].split(',')

            i = int(sI)
            j = int(sJ)
        
            U[i], V[j], cost = gd_update(R[i][j], U[i,:], V[j,:], alpha, lamb)

            cost_sum += cost

        cost_f.append(cost_sum)

    return U, V, cost_f

def dsgd(R, U, V, stratus_number, T, steps, alpha, lamb):

    percent = 0
    current_percent = 0

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

        current_percent = util.calc_progress(T, step+1, current_percent)

        if(current_percent != percent):
            print current_percent
            percent = current_percent

    return U, V

def dgd(R, U, V, stratus_number, T, steps, alpha, lamb):

    percent = 0
    current_percent = 0

    for step in xrange(T):

        list_stratus, list_U, list_V, index_pointer_r, index_pointer_c = stratus.split_matrix(R, U, V, stratus_number, step)

        for i in xrange(stratus_number):

            list_U[i],list_V[i], cost_f = gd(list_stratus[i], list_U[i], list_V[i], steps, alpha, lamb)

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

def extract_train(L):

    L_len = len(L)
    L_half = L_len / 2
    
    ini = randint(0,L_half)
    fin = randint(L_half+1,L_len)

    return L[ini:fin]

    # print L

    # train = []
    # train_index = []

    # train_len = randint(0,len(L))

    # for i in xrange(train_len):

    #     j = randint(0,len(L)-1)

    #     if j not in train_index:

    #         train_index.append(j)
    #         train.append(L[i])

    # return train


