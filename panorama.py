import sys, json ,cplex, numpy, math
import matplotlib.pyplot as plt

I=0
T=4;
Q=3;
K=0;
AP=0;
RBs=0

V=[]
Bm=[]
Rmax=[]
h_i_m=[]
utility=[]
user_mcs=[]
group_mcs=[]
tileWeights = []
utility=[]
debug = 0

def printProblemSolution(problem):

    print("Method used: ", problem.solution.method[problem.solution.get_method()] )
    print("Objective Value: ", problem.solution.get_objective_value() )
    print("Problem status: ", problem.solution.status[problem.solution.get_status()] )
    print("Solution string: ", problem.solution.get_status_string())

    curVar = problem.variables.get_names(0);
    for x in range(problem.variables.get_num()):
        varName = problem.variables.get_names(x)
        varValue = problem.solution.get_values(varName)

        if curVar[0] != varName[0]:
            curVar = varName
            print("\n")
        if(varValue!=0):
            print (varName, "\t", abs(varValue))
            


def getUserLTESolution(problem, user):
    print("\tLTE Solution")
    groupId = -1
    for k in range(K):
        varName = "y." + str(user+1) + "." + str(k+1)
        if problem.solution.get_values(varName) > 0:
            print("\t\t", varName, problem.solution.get_values(varName))
            groupId = k
            
    for k in range(K):
        for t in range(T):
            for q in range(Q):
                varName = "m." + str(user+1) + "." + str(k+1) + "." + str(t+1) + "." + str(q+1)
                varValue = problem.solution.get_values(varName)
                
                groupName = "z." + str(k+1)
                groupRB = problem.solution.get_values(groupName)
                groupRate = groupRB*group_mcs[k]
                if problem.solution.get_values(varName) > 0:
                    print("\t\t group ", k+1, "| mcs ", group_mcs[k], "| RB: ",round(groupRB,2),"| rate", round(groupRate,2), "| tile", t+1, "| quality ", q+1)
                    

def getUserWiFiSolution(problem, user):

    print("\tWiFi Solution")
    for w in range(AP):
        varName = "a." + str(user+1) + "." + str(w+1)
        varValue = problem.solution.get_values(varName)
        userWiFiRate = varValue*h_i_m[user][w]
        
        if problem.solution.get_values(varName) > 0:
            print("\t\t AP :",w+1,"\n\t\t",varName,":", round(varValue,3),"\n\t\t given rate  :",round(userWiFiRate,2), "kbps\n\t\t max rate : ", h_i_m[user][w], "kbps")
            for t in range(T):
                for q in range(Q):
                    varName = "x." + str(user+1) + "." + str(w+1) + "." + str(t+1) + "." + str(q+1)
                    varValue = problem.solution.get_values(varName)
                    if problem.solution.get_values(varName) > 0:
                        print("\t\t\t tile ", t+1, "| quality ",q+1, "| bitrate ", V[q], "kbps")


def printNewSolutionPerUser(problem):

    for i in range(I):
        print("user", i+1)
        getUserLTESolution(problem, i)
        print("\n")
        getUserWiFiSolution(problem, i)
        print("\n")
        #getUserTileSolution(problem, i)



def printOutput(problem):

    print(problem.variables.get_names())
    print(problem.solution.get_values())
    
    #print(problem.solution.get_values(0))
    #print(problem.solution.get_values("x.1.1"))
    

    
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





def generateUniformMcsDistribition(numUsers, numChannels):
    print("Generating uniform MCS distribution for ", int(numUsers) , " users")

    uniformDst = numpy.random.randint(numChannels, size= numUsers)
    #print("\tChannel distribution:\t", uniformDst)
    #plotChannelDistribution(uniformDst)

    return uniformDst
 
 
def generateWifiConnections(I, AP, minWiFiRate, maxWiFiRate):
    global h_i_m
    print("minWiFiRate ", minWiFiRate, "\nmaxWiFiRate ", maxWiFiRate)
    userAP = numpy.random.randint(AP, size=I) # AP index for each user
    userRate = numpy.random.uniform(minWiFiRate, maxWiFiRate, size=I)
    
    for i in range(I):
        tmp = numpy.zeros(AP)
        tmp[ userAP[i] ] = round(userRate[i], 2) # get only two decimal digits
        h_i_m.append(tmp.tolist())
            
    
    #print(h_i_m)
    #print(userRate)
    #print(userAP)
    #exit()
 
    
def readConfiguration():

    global I, AP, K, user_mcs, group_mcs, Rmax, Bm, h_i_m, V, T, Q, RBs

    user_mcs = []
    group_mcs = []
    Rmax = []
    h_i_m = []

    with open('configPanorama.json', 'r') as myfile:
        data = myfile.read()

    obj = json.loads(data)

    print ("System input for json:")
    print (json.dumps(obj, indent=5))

    K = int( obj["Multicast Groups"]["number"] ) # number of multicast groups
    RBs = int( obj["Multicast Groups"]["RBs"] )    # LTE RBs
    I = int( obj["Users"]["number"] )            # number of users


    T  = int( obj["Video"]["tiles"] )             # number of video tiles
    Q  = int( obj["Video"]["representations"])    # video representation per tile
    AP = int( obj["WiFiAPs"]["number"] )         # number of AP

    tmpGroupMcs = obj["Multicast Groups"]["group mcs"]
    group_mcs = [0.0]*K
    Rmax = [0.0]*K
    
    for k in range(K):
        group_mcs[k] = float( tmpGroupMcs[k]*1000*0.6 )/ 1000 # kbps: divide by 100 to make it in kbps
        Rmax[k] = float( group_mcs[k]*RBs )

    if obj["Users"]["mcs distribution"] == "uniform":
        mcs_pos =  generateUniformMcsDistribition(I, K) # users LTE channel distribution
        for i in range(I):
            user_mcs.append( group_mcs[mcs_pos[i]] )
    else:
        print ("Unknown distribution for user mcs")
        exit()
    
    tmpVideo = obj["Video"]["bitrate"]
    V = [0]*Q
    for q in range(Q):
        V[q] = int( tmpVideo[q] )   # rate for each video layer

    Bm = [0.0]*AP
    for m in range(AP):
        Bm[m] = int( obj["WiFiAPs"]["backhaul"] )    # AP backhaul capacity
    
    minWiFiRate = int(obj["WiFiAPs"]["minRate"])
    maxWiFiRate = Bm[0]
    
    #h_i_m = generateWiFiPlane(I, minWiFiRate, maxWiFiRate)
    
    generateWifiConnections(I, AP, minWiFiRate, maxWiFiRate)
    
    #h_i_m= []
    #for i in range(I):
    #    h_i_m.append(  numpy.random.randint(minWiFiRate, Bm[0], size=AP).tolist()  ) # user rate for each AP
    
    #utility = [0.0]*L
    #for l in range(L):
    #    utility[l] = math.log(1 + V[l]/V[0])  # utility for the master problem
    
    for t in range(T):
        tmpValues = []
        for q in range(Q):
            tileUtil = tileWeights[t]*math.log10( V[q] )
            tmpValues.append(tileUtil)
        utility.append(tmpValues)
        

    print("\nUsers: ", I)
    print("\nAPs: ", AP)
    print("\nGroups: ", K)
    print("\nGroup mcs: ", group_mcs)
    print("\nuser mcs: ", user_mcs)
    print("\nBackhaul: ", Bm[0])
    print("\nmaxWiFiRate: ", maxWiFiRate)
    print("\nminWiFiRate: ", minWiFiRate)
    print("\nWifi rate: ",h_i_m)
    print("\nTiles: ", T)
    print("\nQuality per tile: ", Q)
    print("\nBitrate: ", V)
    print("\nUtility: ", utility)
    
    
def generateTileWeights(tiles):
    global tileWeights
    tileWeights = numpy.random.uniform(0.08, 0.99, size=tiles).tolist()
    #tileWeights = [0.4, 0.3, 0.2, 0.1]
    print("\nTiles viewing probability: ",tileWeights)



def generateProblemVariables(problem):
    print("\nGenerating problem variables")
    
    # y.i.k
    for i in range(I):
        for k in range(K):
            varName = "y." + str(i+1) + "." + str(k+1)
            problem.variables.add(obj=[0.0],
                                  lb=[0.0],
                                  ub=[1.0],
                                  types=["C"],
                                  names=[varName])
                                  
    # z.k
    for k in range(K):
        varName = "z." + str(k+1)
        problem.variables.add(obj=[0.0],
                              lb=[0.0],
                              ub=[ float(RBs) ],
                              types=["C"],
                              names=[varName])
          
    
    # a.i.w
    for i in range(I):
        for w in range(AP):
            varName = "a." + str(i+1) + "." + str(w+1)
            problem.variables.add(obj=[0.0],
                                  lb=[0.0],
                                  ub=[1.0],
                                  types=["C"],
                                  names=[varName])
                                  
                                  
    # m.i.k.t.q
    for i in range(I):
        for k in range(K):
            for t in range(T):
                for q in range(Q):
                    varName = "m." + str(i+1) + "." + str(k+1) + "." + str(t+1) + "." + str(q+1)
                    problem.variables.add(obj=[ utility[t][q] ],
                                          lb=[0.0],
                                          ub=[1.0],
                                          types=["B"],
                                          names=[varName])
                                          
    # x.i.w.t.q
    for i in range(I):
        for w in range(AP):
            for t in range(T):
                for q in range(Q):
                    varName = "x." + str(i+1) + "." + str(w+1) + "." + str(t+1) + "." + str(q+1)
                    problem.variables.add(obj=[ utility[t][q] ],
                                          lb=[0.0],
                                          ub=[1.0],
                                          types=["B"],
                                          names=[varName])

                                  
    if debug ==1:
        print("\tNames: ", problem.variables.get_names())
        print("\tTypes: ", problem.variables.get_types())
        print("\t\nObj Function: ", problem.objective.get_linear())

    #exit()


def generateProblemConstraints(problem):
    print("Generating problem constraints")
    
    # y.i.k * c.k' <= c.i
    for i in range(I):
        for k in range(K):
            theVars = []
            theCoefs = []
            
            tmpName = "y."+ str(i+1) + "." + str(k+1)
            theVars.append(tmpName)
            theCoefs.append(group_mcs[k])
            
            if debug == 1:
                print ("\t\t",theVars, theCoefs, " <= ", user_mcs[i])
                
            problem.linear_constraints.add(lin_expr=[cplex.SparsePair(theVars, theCoefs)],
                                           senses = ["L"],
                                           rhs = [user_mcs[i]])


    # sum_k(y.i.k) <= 1
    for i in range(I):
        theVars = []
        theCoefs = []
        
        for k in range(K):
            tmpName = "y."+ str(i+1) + "." + str(k+1)
            theVars.append(tmpName)
            theCoefs.append(1)

        if debug == 1:
            print ("\t\t",theVars, theCoefs," <= 1")
            
        problem.linear_constraints.add(lin_expr = [cplex.SparsePair(theVars, theCoefs)],
                                       senses = ["L"],
                                       rhs = [1.0])



    # sum_k(z.k) <= RBs
    theVars=[]
    theCoefs=[]
    for k in range(K):
        tmpName = "z." + str(k+1)
        theVars.append(tmpName)
        theCoefs.append(1)
        
    if debug == 1:
        print ("\t\t",theVars, theCoefs, " <= ", RBs)
        
    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(theVars, theCoefs)],
                                   senses=["L"],
                                   rhs=[float(RBs)])
                                   
      
    
    # sum_i(a.i.w*h.i.m) <= Bm
    for w in range(AP):
        theVars = []
        theCoefs = []
        for i in range(I):
            tmpName = "a." +str(i+1) + "." + str(w+1)
            theVars.append(tmpName)
            theCoefs.append(h_i_m[i][w])
        
        if debug == 1:
            print("\t\t", theVars, theCoefs, "<=",Bm[w])
            
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(theVars, theCoefs)],
                                       senses=["L"],
                                       rhs=[ Bm[w] ])



    # sum_i(a.i.w) <= 1
    for w in range(AP):
        theVars = []
        theCoefs = []
        for i in range(I):
            tmpName = "a." +str(i+1) + "." + str(w+1)
            theVars.append(tmpName)
            theCoefs.append(1)
            
        if debug == 1:
            print("\t\t", theVars, theCoefs, "<= 1")
            
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(theVars, theCoefs)],
                                        senses=["L"],
                                        rhs=[ 1.0 ])
    
    
    
    # sum_t( sum_q( m.i.k.t.q*b.t.q ) ) - c.k*z.k  <= 0
    for i in range(I):
        for k in range(K):
            theVars = []
            theCoefs = []
            for t in range(T):
                for q in range(Q):
                    tmpName = "m." + str(i+1) + "." + str(k+1) + "." + str(t+1) + "." + str(q+1)
                    theVars.append(tmpName)
                    theCoefs.append( V[q] )
            
            tmpName  = "z." + str(k+1)
            theVars.append(tmpName)
            theCoefs.append(-group_mcs[k])
            
            if debug == 1:
                print("\t\t", theVars, theCoefs, "<= 0")
                
            problem.linear_constraints.add(lin_expr=[cplex.SparsePair(theVars, theCoefs)],
                                            senses=["L"],
                                            rhs=[0.0])
                
    

    
    # sum_i( sum_w( x.t.w.t.q*b.t.q )) -h.i.w*a.i.w <=0
    for i in range(I):
        for w in range(AP):
            theVars = []
            theCoefs = []
            for t in range(T):
                for q in range(Q):
                    tmpName = "x." + str(i+1) + "." + str(w+1) + "." + str(t+1) + "." + str(q+1)
                    theVars.append(tmpName)
                    theCoefs.append( V[q] )
            
            tmpName = "a." + str(i+1) + "." + str(w+1)
            theVars.append(tmpName)
            theCoefs.append(-h_i_m[i][w])
            
            if debug == 1:
                print("\t\t", theVars, theCoefs, "<= 0")
                
            problem.linear_constraints.add(lin_expr=[cplex.SparsePair(theVars, theCoefs)],
                                            senses=["L"],
                                            rhs=[0.0])
                                            
       

    # sum_k( sum_q( m.i.k.t.q) ) + sum_w( sum_q( x.i.w.t.q) ) = 1
    for i in range(I):
        for t in range(T):
            theVars = []
            theCoefs = []
            for k in range(K):
                for q in range(Q):
                    tmpName = "m." + str(i+1) + "." + str(k+1) + "." + str(t+1) + "." + str(q+1)
                    theVars.append(tmpName)
                    theCoefs.append(1)
                    
            for w in range(AP):
                for q in range(Q):
                    tmpName = "x." + str(i+1) + "." + str(w+1) + "." + str(t+1) + "." + str(q+1)
                    theVars.append(tmpName)
                    theCoefs.append(1)
                    
            
            if debug == 1:
                print("\t\t", theVars, theCoefs, "= 1")
                
            problem.linear_constraints.add(lin_expr=[cplex.SparsePair(theVars, theCoefs)],
                                            senses=["E"],
                                            rhs=[1.0])
                                            
    
    # m.i.k.t.q <= y.i.k
    for i in range(I):
        for k in range(K):
            for t in range(T):
                for q in range(Q):
                    theVars = []
                    theCoefs = []
                    
                    tmpName = "m." + str(i+1) + "." + str(k+1) + "." + str(t+1) + "." + str(q+1)
                    theVars.append(tmpName)
                    theCoefs.append(1)
                    
                    tmpName = "y." + str(i+1) + "." + str(k+1)
                    theVars.append(tmpName)
                    theCoefs.append(-1)
                    
                    if debug == 1:
                         print("\t\t", theVars, theCoefs, "<= 0")
                         
                    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(theVars, theCoefs)],
                                                    senses=["L"],
                                                    rhs=[0.0])
                                    

def optimalPanoramicVideo():

    problem = cplex.Cplex()
    problem.objective.set_sense(problem.objective.sense.maximize)
    problem.parameters.preprocessing.presolve.set(problem.parameters.preprocessing.presolve.values.off)
    #problem.parameters.timelimit.set(1000) # ~16 minutes
    
    #problem.parameters.mip.tolerances.mipgap.set(0.05)
    
    generateProblemVariables(problem)
    generateProblemConstraints(problem)
    
    problem.solve()
    printProblemSolution(problem)
    #printNewSolutionPerUser(problem)
    
    #printOutput(problem)


if __name__ == "__main__":

    print("Panoramic Multicast Groups ( -- mrmo360 -- )")
    
    generateTileWeights(32)
    readConfiguration()
    optimalPanoramicVideo()
