import cplex
import json
import numpy
import math
import scipy.io
import time
import sys
import matplotlib.pyplot as plt
#import seaborn as sns


# System model global variables
I=0
K=0
L=0
T=0
AP=0
V=[]
Bm=[]
h_i_m=[]
Rmax=[]
utility=[]
user_mcs=[]
group_mcs=[]
GWiFi_penalty=0

quality = 0

def plotChannelDistribution(values):
    print("Plotting channel distribution")
    
    mcsFreq = []
    mcsValues = []
    for item in values:
        if item not in mcsValues:
            mcsValues.append(item)

    for mcs in mcsValues:
        cnt = 0
        for val in values:
            if mcs == val:
                cnt+=1
        mcsFreq.append(cnt)

    plt.close()
    plt.bar(mcsValues, mcsFreq)
    plt.xticks(numpy.arange(0, 15, step=1))
    plt.title('MCS distribution')
    plt.ylabel('Number of Users')
    plt.xlabel('MCS index')
    plt.savefig('results/tests/lteMCSDistribution.eps')


def generateUniformMcsDistribition(numUsers, numChannels):
    print("Generating uniform MCS distribution for ", int(numUsers) , " users")
    
    uniformDst = numpy.random.randint(numChannels, size= numUsers)
    #print("\tChannel distribution:\t", uniformDst)
    #plotChannelDistribution(uniformDst)
    
    return uniformDst

def generateWiFiPlane(numUsers, minRate, maxRate):
    print("Generating eNB WiFi plane")
    
    wifiRate = []
    ap_range = 250
    ap_distance = 300
    grid_x = 5
    grid_y = 2
    numAps = grid_x*grid_y
    ap_xCoord = []
    ap_yCoord = []
    x_coord = 300
    
    # creating the grid topology for WiFi APs
    for x_ in range(grid_x):
        y_coord = 300
        for y_ in range(grid_y):
            ap_xCoord.append(x_coord)
            ap_yCoord.append(y_coord)
            y_coord = y_coord + ap_distance
        x_coord = x_coord + ap_distance

    # uniformly distributing users in the grid
    user_xCoord = []
    user_yCoord = []
    user_xCoord = numpy.random.randint(80, (grid_x+1)*ap_distance-80, size = numUsers).tolist()
    user_yCoord = numpy.random.randint(80, (grid_y+1)*ap_distance-80, size = numUsers).tolist()

    # associate users to AP and assign a uniform random trhoughput
    for ue in range(numUsers):
        apList = []
        for ap in range(numAps):
            x_dist = (abs(user_xCoord[ue] - ap_xCoord[ap]))**2
            y_dist = (abs(user_yCoord[ue] - ap_yCoord[ap]))**2
            distance = math.sqrt(x_dist+y_dist)
            if distance <= ap_range:
                rate = numpy.random.randint(minRate, maxRate)
                apList.append(rate)
            else:
                apList.append(0)

        wifiRate.append(apList)

    #print(wifiRate)

    plt.close()
    plt.scatter(user_xCoord, user_yCoord, marker="1", label="users")
    plt.scatter(ap_xCoord, ap_yCoord, marker="D", label="WiFi AP")
    plt.xlabel("meters")
    plt.ylabel("meters")
    plt.grid()
    plt.legend()

    #plt.show()
#exit()

    return wifiRate


def readSystemModelInput(customBm, customUsers):
    
    global I, AP, K, L, user_mcs, group_mcs, Rmax, Bm, h_i_m, utility, V, T
    
    user_mcs = []
    group_mcs = []
    Rmax = []
    h_i_m = []
    utility = []
    
    with open('input.json', 'r') as myfile:
        data = myfile.read()

    obj = json.loads(data)

    print ("System input for json:")
    print (json.dumps(obj, indent=5))

    K = int( obj["Multicast Groups"]["number"] ) # number of multicast groups
    T = int( obj["Multicast Groups"]["RBs"] )    # LTE RBs
    I = int( obj["Users"]["number"] )            # number of users

    I = int(customUsers)

    L = int( obj["Video"]["layers"] )            # number of video layers
    AP = int( obj["WiFiAPs"]["number"] )         # number of AP

    tmpGroupMcs = obj["Multicast Groups"]["group mcs"]
    group_mcs = [0.0]*K
    Rmax = [0.0]*K
        
    for k in range(K):
        group_mcs[k] = float( tmpGroupMcs[k]*1000*0.6 )
        Rmax[k] = float( group_mcs[k]*T )

    if obj["Users"]["mcs distribution"] == "uniform":
        mcs_pos =  generateUniformMcsDistribition(I, K) # users LTE channel distribution
        for i in range(I):
            user_mcs.append( group_mcs[mcs_pos[i]] )
    else:
        print ("Unknown distribution for user mcs")
        exit()
        
    tmpVideo = obj["Video"]["bitrate"]
    V = [0]*L
    for l in range(L):
        V[l] = int( tmpVideo[l] )   # rate for each video layer

    Bm = [0.0]*AP
    for m in range(AP):
        Bm[m] = int( obj["WiFiAPs"]["backhaul"] )    # AP backhaul capacity
        Bm[m] = int(customBm)
        
    minWiFiRate = int(obj["WiFiAPs"]["minRate"])
    maxWiFiRate = customBm
        
    h_i_m = generateWiFiPlane(I, minWiFiRate, maxWiFiRate)
        
    '''
    h_i_m= []
    for i in range(I):
    h_i_m.append(  numpy.random.randint(minWiFiRate, Bm[0], size=AP).tolist()  ) # user rate for each AP
    '''
        
    utility = [0.0]*L
    for l in range(L):
        utility[l] = math.log(1 + V[l]/V[0])  # utility for the master problem

    print("\nUsers: ", I)
    print("\nVideo: ", V)
    print("\ngroups: ", K)
    print("\nLayers: ", L)
    print("\nAPs: ", AP)
    print("\nBackhaul: ", Bm[0])
    print("\nmaxWiFiRate: ", maxWiFiRate)
    print("\nminWiFiRate: ", minWiFiRate)
    print("\nuser mcs: ", user_mcs)
    print("\nWifi rate: ",h_i_m)


def createENEMG_Variables(problem):
    
    alpha_lte = 51.97/10**6
    beta_lte  = 1288.04
    
    alpha_wifi = 137.01/10**6
    beta_wifi  = 132.86
    
    # x.i.k
    for i in range(I):
        for k in range(K):
            varName = "x." + str(i+1) + "." + str(k+1)
            problem.variables.add(obj=[beta_lte],
                                  lb=[0.0],
                                  ub=[1.0],
                                  types=["B"],
                                  names=[varName])

    # z.k
    for k in range(K):
        varName = "z." + str(k+1)
        problem.variables.add(obj=[0.0],
                              lb=[0.0],
                              ub=[ float(T) ],
                              types=["C"],
                              names=[varName])
            
    # r.i.k
    for i in range(I):
        for k in range(K):
            varName = "r." + str(i+1) + "." + str(k+1)
            problem.variables.add(obj=[ alpha_lte ],
                                  lb=[0.0],
                                  ub=[ Rmax[k] ],
                                  types=["C"],
                                  names=[varName])

    # y.i.m
    for i in range(I):
        for m in range(AP):
            varName = "y." + str(i+1) + "." + str(m+1)
            problem.variables.add(obj=[beta_wifi],
                                  lb=[0],
                                  ub=[1],
                                  types=["B"],
                                  names=[varName])

    # q.i.m
    for i in range(I):
        for m in range(AP):
            varName = "q." + str(i+1) + "." + str(m+1)
            problem.variables.add(obj=[ alpha_wifi*h_i_m[i][m] ],
                                  lb=[0.0],
                                  ub=[1.0],
                                  types=["C"],
                                  names=[varName])

    #print("\tVariables: ", problem.variables.get_names())
    #print("\tVariables: ", problem.variables.get_types())
    #print("\t\nObj Function: ", problem.objective.get_linear())
    #exit()

def createENEMG_Constraints(problem):
    
    # sum(x.i.k) <= 1
    for i in range(I):
        theVars = []
        theCoefs = []
        
        for k in range(K):
            tmpName = "x."+ str(i+1) + "." + str(k+1)
            theVars.append(tmpName)
            theCoefs.append(1)
    
        print ("\t\t",theVars, theCoefs," <= 1")
        problem.linear_constraints.add(lin_expr = [cplex.SparsePair(theVars, theCoefs)],
                                       senses = ["L"],
                                       rhs = [1.0])


    # sum(y.i.m) <= 1
    for i in range(I):
        theVars = []
        theCoefs = []
        
        for m in range(AP):
            tmpName = "y." + str(i+1) + "." + str(m+1)
            theVars.append(tmpName)
            theCoefs.append(1)
            
        print ("\t\t",theVars, theCoefs, " <= 1")
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(theVars, theCoefs)],
                                       senses = ["L"],
                                       rhs = [1.0])


    # c.k' * x.i.k <= c.i
    for i in range(I):
        for k in range(K):
            theVars = []
            theCoefs = []
            
            tmpName = "x."+ str(i+1) + "." + str(k+1)
            theVars.append(tmpName)
            theCoefs.append(group_mcs[k])
                
            print ("\t\t",theVars, theCoefs, " <= ", user_mcs[i])
            problem.linear_constraints.add(lin_expr=[cplex.SparsePair(theVars, theCoefs)],
                                           senses = ["L"],
                                           rhs = [user_mcs[i]])


    # sum(h.i.m*q.i.m) <= Bm
    for m in range(AP):
        theVars = []
        theCoefs = []
        
        for i in range(I):
            tmpName = "q." + str(i+1) + "." + str(m+1)
            theVars.append(tmpName)
            theCoefs.append(h_i_m[i][m])
            
        print ("\t\t",theVars, theCoefs, " <= ", Bm[m])
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(theVars, theCoefs)],
                                       senses=["L"],
                                       rhs = [ Bm[m] ])


    # sum(q.i.m)<=1
    for m in range(AP):
        theVars = []
        theCoefs = []
        
        for i in range(I):
            tmpName = "q." + str(i+1) + "." + str(m+1)
            theVars.append(tmpName)
            theCoefs.append(1)
            
        print ("\t\t",theVars, theCoefs, " <= ", 1)
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(theVars, theCoefs)],
                                       senses=["L"],
                                       rhs = [1.0])


    # sum(sum(z.k)) <= T
    theVars=[]
    theCoefs=[]
    for k in range(K):
        tmpName = "z." + str(k+1)
        theVars.append(tmpName)
        theCoefs.append(1)
        
    print ("\t\t",theVars, theCoefs, " <= ", T)
    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(theVars, theCoefs)],
                                   senses=["L"],
                                   rhs=[float(T)])



    # h.i.m*q.i.m  -B(m)*y.i.m <= 0
    for i in range(I):
        for m in range(AP):
            theVars=[]
            theCoefs=[]
            
            tmpName = "q." + str(i+1) + "." + str(m+1)
            theVars.append(tmpName)
            theCoefs.append(h_i_m[i][m])
            
            tmpName = "y."+ str(i+1) + "." + str(m+1)
            theVars.append(tmpName)
            theCoefs.append(-Bm[m])
            
            print ("\t\t",theVars, theCoefs, " <= 0")
            problem.linear_constraints.add(lin_expr=[cplex.SparsePair(theVars, theCoefs)],
                                           senses=["L"],
                                           rhs=[ 0.0 ])

    # -sum(r.i.k) - sum(h.i.m*q.i.m) <= -V[l]
    for i in range(I):
        theVars=[]
        theCoefs=[]
        
        for k in range(K):
            tmpName = "r." + str(i+1) + "." + str(k+1)
            theVars.append(tmpName)
            theCoefs.append(-1.0)
            
        for m in range(AP):
            tmpName = "q." + str(i+1) + "." + str(m+1)
            theVars.append(tmpName)
            theCoefs.append(-h_i_m[i][m])
            
        print ("\t\t",theVars, theCoefs, " <= 0")
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(theVars, theCoefs)],
                                       senses=["L"],
                                       rhs=[-V[quality]])


    # r.i.k - c(k)*z.k <= 0
    for i in range(I):
        for k in range(K):
            theVars=[]
            theCoefs=[]
            
            tmpName = "r." + str(i+1) + "." + str(k+1)
            theVars.append(tmpName)
            theCoefs.append(1)

            tmpName = "z." + str(k+1)
            theVars.append(tmpName)
            theCoefs.append( -group_mcs[k] )

            print ("\t\t",theVars, theCoefs, " <= 0")
            problem.linear_constraints.add(lin_expr=[cplex.SparsePair(theVars, theCoefs)],
                                           senses=["L"],
                                           rhs=[ 0.0 ])

                    # r.i.k - Rmax(k)*x.i.k <= 0
    for i in range(I):
        for k in range(K):
            theVars=[]
            theCoefs=[]
            
            tmpName = "r." + str(i+1) + "." + str(k+1)
            theVars.append(tmpName)
            theCoefs.append(1)
            
            tmpName = "x." + str(i+1) + "." + str(k+1)
            theVars.append(tmpName)
            theCoefs.append(-Rmax[k])
            
            print ("\t\t",theVars, theCoefs, " <= 0")
            problem.linear_constraints.add(lin_expr=[cplex.SparsePair(theVars, theCoefs)],
                                           senses=["L"],
                                           rhs=[ 0.0 ])
    exit()

#print ("\tNumber of Master Constraints", problem.linear_constraints.get_num())
#print ("\tRhs:", problem.linear_constraints.get_rhs())


def ENEMG(curGap):
    
    problem = cplex.Cplex()
    problem.objective.set_sense(problem.objective.sense.minimize)
    problem.parameters.preprocessing.presolve.set(problem.parameters.preprocessing.presolve.values.off)
    problem.parameters.timelimit.set(1000) # ~16 minutes
    
    createENEMG_Variables(problem)
    createENEMG_Constraints(problem)
    
    #problem.set_results_stream(None)
    problem.set_log_stream(None)
    problem.set_error_stream(None)
    problem.set_warning_stream(None)
    
    
    '''
        # Optimal gap tolerance
        #master.parameters.mip.tolerances.mipgap.set(0.1)
        # absolute gap tolerance
        #master.parameters.mip.tolerances.absmipgap.set(0.01)
        '''
    #problem.parameters.mip.tolerances.mipgap.set(curGap)

    problem.write("ENEMG.lp")

    
if __name__ == "__main__":
    
    curUsers = int(sys.argv[1])
    curGap = float(sys.argv[2])
    curBm = 20000000

    readSystemModelInput(curBm, curUsers)
    ENEMG(curGap)
