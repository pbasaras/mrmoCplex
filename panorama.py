import sys, json ,cplex, numpy, math
import matplotlib.pyplot as plt
from scipy.optimize import *
from numpy import *
import pandas as pd
import statistics


I=0
T=4;
Q=3;
K=0;
AP=0;
RBs=0
AUXILIARY = 10000

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


optGap = 0.05

RESULTS = []



# crossLayer Optimization : globecom_2018
MaxSE = 0
userGroups = []
minRBForLowestGroup=0
#groupResources = []
#groupTiles = []

def printProblemSolution(problem):

    print("Method used: ", problem.solution.method[problem.solution.get_method()] )
    print("Objective Value: ", problem.solution.get_objective_value() )
    print("Problem status: ", problem.solution.status[problem.solution.get_status()] )
    print("Solution string: ", problem.solution.get_status_string())

    return

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

    for k in range(K):
        for t in range(T):
            for q in range(Q):
                varName = "y." + str(k+1) + "." + str(t+1) + "." + str(q+1)
                varValue = problem.solution.get_values(varName)
                if problem.solution.get_values(varName) > 0:
                    print(varName,varValue)
                    
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

    global I, AP, K, user_mcs, group_mcs, Rmax, Bm, h_i_m, V, T, Q, RBs, utility

    user_mcs = []
    group_mcs = []
    Rmax = []
    h_i_m = []
    utility = []

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
    tileWeights = numpy.ones(tiles).tolist()
    print("\nTiles viewing probability: ",tileWeights)



def generateProblemVariables(problem):
    print("\nGenerating problem variables")
    
    # y.k.t.q
    for k in range(K):
        for t in range(T):
            for q in range(Q):
                varName = "y." + str(k+1) + "." + str(t+1) + "." + str(q+1)
                problem.variables.add(obj=[0.0],
                                        lb=[0.0],
                                        ub=[1.0],
                                        types=["B"],
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
    
    
    
    # sum_t( sum_q( y.k.t.q*b.t.q ) ) <= z.k*c.k
    for k in range(K):
        theVars = []
        theCoefs = []
        for t in range(T):
            for q in range(Q):
                tmpName = "y." + str(k+1) + "." + str(t+1) + "." +str(q+1)
                theVars.append(tmpName)
                theCoefs.append(V[q])
                
        tmpName  = "z." + str(k+1)
        theVars.append(tmpName)
        theCoefs.append(-group_mcs[k])
        
        if debug == 1:
            print("\t\t", theVars, theCoefs, "<= 0")
        
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(theVars, theCoefs)],
                                        senses=["L"],
                                        rhs=[0.0])
    
    
    # sum_i( m.i.k.t.q ) - y.k.t.q <= 0
    for k in range(K):
        for t in range(T):
             for q in range(Q):
                theVars = []
                theCoefs = []
                for i in range(I):
                    tmpName = "m." + str(i+1) + "." + str(k+1) + "." + str(t+1) + "." +str(q+1)
                    theVars.append(tmpName)
                    theCoefs.append(1)
                
                tmpName = "y." + str(k+1) + "." + str(t+1) + "." + str(q+1)
                theVars.append(tmpName)
                theCoefs.append(-AUXILIARY)

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
    
    
    
    #sum_q( m.i.k.t.q*c.k ) <= c.i
    for i in range(I):
        for k in range(K):
            for t in range(T):
                theVars = []
                theCoefs = []
                for q in range(Q):
                    tmpName = "m." + str(i+1) + "." + str(k+1) + "." + str(t+1) + "." + str(q+1)
                    theVars.append(tmpName)
                    theCoefs.append(group_mcs[k])
                    
                if debug == 1:
                    print("\t\t", theVars, theCoefs, "<=", user_mcs[i])
                    
                problem.linear_constraints.add(lin_expr=[cplex.SparsePair(theVars, theCoefs)],
                                                senses=["L"],
                                                rhs=[user_mcs[i]])
    
    

def optimalPanoramicVideo(competitors):

    problem = cplex.Cplex()
    problem.objective.set_sense(problem.objective.sense.maximize)
    problem.parameters.preprocessing.presolve.set(problem.parameters.preprocessing.presolve.values.off)
    #problem.parameters.timelimit.set(1000) # ~16 minutes
    problem.parameters.mip.tolerances.mipgap.set(optGap)
    
    generateProblemVariables(problem)
    generateProblemConstraints(problem)
    
    problem.solve()
   
    #printNewSolutionPerUser(problem)
    printProblemSolution(problem)
    #printOutput(problem)
    print("PUMA: Obj Value: ", problem.solution.get_objective_value())
    RESULTS[4]+=problem.solution.get_objective_value()
    competitors[4].values.append(problem.solution.get_objective_value())
    
##############################################################################################################
##############################################################################################################


def getTileUtility(rate):
    return numpy.log10(rate)

def groupSplitHeuristic():
    global MaxSE, userGroups
    
    print("prev MaxSE", MaxSE)
    
    lastGroup = len(userGroups)-1 # always to the split to the last group
    print("User MCS List : ", userGroups[lastGroup])
    currentMCSList = set(userGroups[lastGroup])
    uniqueMCSList  = list(currentMCSList)
    uniqueMCSList.sort()    # always extract the unique mcs values in the last group
    print("Unique MCS list : ",uniqueMCSList)
    
    minMCS = min(uniqueMCSList) # get the minMCS which will me the SE for the left group of the split
    outputMCS = minMCS
    print("Min MCS ", minMCS)
    curUtil =0
    potentialGroupSplits = len(uniqueMCSList)-1 # the potential splits will be one less than the total uniqeu mcs indexes
    for i in range(potentialGroupSplits):
        splitMCS = uniqueMCSList[i+1] # start from the second unique mcs index for the first split, and move to subsequent ones
        leftUsers = 0
        rightUsers = 0
        for mcs in  userGroups[lastGroup]:
            if mcs < splitMCS:
                leftUsers +=1
            else:
                rightUsers +=1
           
        leftUtility  = leftUsers*minMCS
        rightUtility = rightUsers*splitMCS
        totalUtility = leftUtility + rightUtility
        
        if curUtil < totalUtility:
            curUtil = totalUtility
            outputMCS = splitMCS
            
        print("split mcs", splitMCS, "l: ", leftUsers,", r: ", rightUsers)
        print("LEFT Util", leftUtility)
        print("RIGHT Util", rightUtility)
        print("TOTAL Util", totalUtility)
        print("\n")
    
    print("Selected split ", outputMCS)
    
    tmpList = []
    numItems = 0
    for item in userGroups[lastGroup]:
        if item >= outputMCS:
            tmpList.append(item)
            numItems += 1
    
    for i in range(numItems):
        userGroups[lastGroup].pop()
            
    #print(userGroups)
    userGroups.append(tmpList)
    print(userGroups)
            

def crossLayergrouping():
    print("\nCrossLayer grouping")
    global MaxSE, userGroups
    
    if MaxSE == 0:
        MaxSE = min(user_mcs)*I
        userGroups.append(sorted(user_mcs))
        print("\t", userGroups)
    else:
        groupSplitHeuristic()


lagGroups = 0
lagCon = []
laSol = []

def getMaxLagrangeSolution(groups):
     print(laSol)

def lagrangeFunction(z):

    global lagGroups, lagCon
    
    numEquations = 1 + lagGroups + lagGroups # lambda + z.g + Nrb.g
    F = empty((numEquations))
    
    eq = 0
    F[eq] = -RBs # eq 1
    for i in range(lagGroups):
        F[eq] =  F[eq] + z[lagGroups+i+1]

    eq += 1
    for i in range(lagGroups):
        F[eq]= z[i+1]*z[lagGroups+i+1] # eq 3
        eq+=1
        
    for i in range(lagGroups):
        F[eq] = z[0]*z[lagGroups+i+1] -lagCon[i] # eq 16
        eq +=1

    return F


def crossLayerResourceAllocation(groups):
    print("CrossLayer Resource Allocation, Number of groups ", len(groups))
    
    global lagGroups, lagCon, laSol
    
    laSol = []
    lagCon = []
    lagGroups = len(groups)
    
    for g in groups:
        #g.printGroupData()
        #lagCon.append( g.Ag*(g.RBs-g.residualRB) )
        lagCon.append(g.Ag*g.RBs)
        #print("id: ", g.id, "\n\tlaCon: ", lagCon[g.id-1], "\n\tAg: ", g.Ag)

    numVariables = 1+2*lagGroups
    zGuess=numpy.zeros(numVariables)
    z = fsolve(lagrangeFunction, zGuess)
    
    pos = 1 + lagGroups
    for g in groups:
        g.RBs = math.floor(z[pos])
        g.residualRB = math.floor(z[pos])
        g.maxRate= g.mcs*g.RBs
        pos = pos+1
        #g.printGroupData()
    
    print("=================== NEW allocated resource Blocks =================")
    print(z)


def curveFitFunction(x,a,b):
    return a*numpy.log10(b*x)


    
class Group:
    def __init__(self, id, RBs, mcs, users):
        self.id = id
        self.Utility = 0
        self.RBs = RBs
        self.residualRB = self.RBs
        self.mcs = mcs
        self.maxRate = self.RBs*self.mcs
        self.users = users
        self.pairs = []
        self.Ag = 0.0
        self.Bg = 0.0
        self.Ctm = []
        self.effUtil = []
        self.tiles = [] # the tile qualities the group has checked for tranmsission.
        self.tilesQualityInit()
        self.effectiveUtilityInit()

    # init all tiles and all qualities to -1
    def tilesQualityInit(self):
        self.tiles = []
        for t in range(T):
            self.tiles.append(numpy.ones(Q).tolist())
        
        for t in range(T):
            for q in range(Q):
                self.tiles[t][q] = -1*self.tiles[t][q]
    
    
    # set the effective utility as eq. (19)
    def effectiveUtilityInit(self):
        for t in range(T):
            tmp = []
            for q in range(Q):
                if q == 0:
                    effectiveUtil = getTileUtility(V[q])
                    tmp.append( effectiveUtil )
                else:
                    effectiveUtil = getTileUtility(V[q]) - getTileUtility(V[q-1])
                    tmp.append( effectiveUtil )
            self.effUtil.append(tmp)
    
    def curvefiting(self):
        print("curve fitting")
        x_data = []
        y_data = []
        
        
        print(self.pairs, self.RBs, self.residualRB)
        
        if len(self.pairs) <=1:
            return
        
        for d in range(len(self.pairs)):
            x_data.append(self.pairs[d][0])
            y_data.append(self.pairs[d][1])
        
        x_data = numpy.array(x_data)
        y_data = numpy.array(y_data)
        
        #print(x_data)
        #print(y_data)
        #print(self.pairs)
        popt, pcov = curve_fit(curveFitFunction, x_data, y_data, maxfev=100000)
        
        
        self.Ag = popt[0]
        self.Bg = popt[1]
        #print(popt)
        
        f= popt[0]*numpy.log10(x_data*popt[1])
        
        #fit = numpy.polyfit(x_data, y_data, 1)
        #f = fit[0]*x_data-fit[1]
        
        plt.plot(x_data, y_data, "o")
        plt.plot(x_data,f, label='fit params: a=%5.3f, b=%5.3f' % tuple(popt))
        plt.xlabel('RBs')
        plt.ylabel('Utility')
        plt.legend()
        fileName = "group_" + str(self.id) +".eps"
        plt.savefig(fileName, format="eps")
        #exit()
    
    def printGroupData(self):
        print("-------- Group Data --------\nId: ", self.id,"\nGroup Utility: ", self.Utility,"\neffUtil: ", self.effUtil, "\nUsers: ", self.users,"\nRBs: ", self.RBs,"\nResidual RB: ", self.residualRB ,"\nmcs: ", self.mcs, "\nrate: ", self.maxRate,"kbps\npais: ", self.pairs,"\nAg: ", self.Ag,"\nBg: ", self.Bg)
        print("Cost Matrix: ", self.Ctm)
        print("Tile Qualities: ", self.tiles)

        

# den prepei na einai v[q] - v[q_selected]?
def generateTileRepresentationCostMatrixForLowestGroup(groups):

    for t in range(T):
        tmpCost = []
        for q in range(Q):
            if q == 0:
                tmpCost.append(V[q])
            else:
                tmpCost.append(V[q] - V[q-1])
                
        groups[0].Ctm.append(tmpCost)
            

def getSelectedTileQualityOfGroup(groups, idx, t):
    
    selectedQuality = 0
    for q in range(Q):
        if groups[idx].tiles[t][q] != -1:
            selectedQuality = q

    return selectedQuality


def updateTileRepresentationCostMatrix(groups, idx):
    print("\n\n\n\nupdating cost matrix for group idx ", idx)
    
    for t in range(T):
        tmpCost = []
        for q in range(Q):
            prevGroupIdx = idx -1
            prevTileQuality = getSelectedTileQualityOfGroup(groups, prevGroupIdx, t)
            #print("Tile ", t, " Quality ", prevTileQuality)
            if q <= prevTileQuality:
                tmpCost.append(0)
            elif q == prevTileQuality + 1:
                tmpCost.append(V[q])
            elif q >= prevTileQuality+ 2:
                tmpCost.append(V[q] - tmpCost[q-1])
        
        groups[idx].Ctm.append(tmpCost)
     

def updateTileSelectionMatrix(groups, idx):
    
    for t in range(T):
        for q in range(Q):
            groups[idx].tiles[t][q] = groups[idx-1].tiles[t][q]
    
    
# this function will NOT allocate Tiles as in the iters we will select the nxt tile.
def initializeRateForLowestRepresentation(groups):

    totalRate = 0
    for g in groups:
        for t in range(T):
            g.tiles[t][0] = 0 # tile 0, the lowest representation tile is selected
            if g.id == 1: # allocate the necesary rate of the lower representation only to the first group
                totalRate += V[0]
                g.Utility += getTileUtility(V[0])*tileWeights[t]
    
    consumedRB = math.ceil(totalRate/groups[0].mcs)
    print("INITIALIZING FIRST GROUP\n\tNecessary Rate: ", totalRate, "\tmcs: ", groups[0].mcs,"\tAvailable RB ", groups[0].RBs ,"\tRB needed ", consumedRB)
    
    #groups[0].Utility*=len(groups[0].users)
    groups[0].residualRB -= consumedRB
    groups[0].pairs.append([consumedRB, groups[0].Utility])
    #groups[0].printGroupData()
    
    if consumedRB > groups[0].RBs:
        print("INFEASIBLE RESOURCE ALLOCATION\n\tconsumbedRB ", consumedRB,"\n\tAvailableRB ", g.RBs)
        exit()
        
    return consumedRB
    

def allocateBestTile(groups, idx):
        
    print("\n\nAllocating best tile for group idx", idx)
    
    foundBetterTile = False
    
    bestTileIdx = 0
    bestTileRB  = 0
    bestTileValue = 0
    bestTileQuality = 0
    
    for t in range(T):
        for q in range(Q):
            if groups[idx].tiles[t][q] == -1 and groups[idx].Ctm[t][q] != 0: # that means that this tile has not been allocated to the group yet
            
                #curValue =  getTileUtility(t,V[q])/groups[idx].Ctm[t][q]
                curValue = (groups[idx].effUtil[t][q]*tileWeights[t])/groups[idx].Ctm[t][q]
                neededRB = math.ceil(groups[idx].Ctm[t][q]/groups[idx].mcs)
                
                if curValue > bestTileValue and groups[idx].residualRB >= neededRB:
                    foundBetterTile = True
                    bestTileValue = groups[idx].effUtil[t][q]*tileWeights[t]
                    bestTileIdx = t
                    bestTileQuality = q
                    bestTileRB = neededRB
                    
    if foundBetterTile:
        groups[idx].tiles[bestTileIdx][bestTileQuality] = bestTileQuality
        groups[idx].residualRB -= bestTileRB
        print("\nBest tile: ", bestTileIdx, " at quality ", bestTileQuality, " with rate ", V[bestTileQuality], "kbps and additional RB ", bestTileRB, " utility increase ", bestTileValue)
        #groups[idx].printGroupData()
    else:
        print("No more tiles to check or no RBs left")
        bestTileIdx = -1
        bestTileQuality = -1
        bestTileRB = -1
        bestTileValue = -1
    
    
    return [bestTileIdx, bestTileQuality, bestTileRB, bestTileValue]

    
def crossLayerRateSelection(groups):
    print("CrossLayer Rate Selection, Number of groups: ", len(groups))
    
    for g in groups:
        
        gIdx = g.id -1
        #curUtil = 0
        curRate = 0
        
        # -- ADDED
        g.tilesQualityInit()
        g.Ctm = []
        g.pairs = []
        g.Utility = 0
        
        print("\n\nTile allocation for group, ", g.id, "\tRB ", g.RBs,"\tresidualRB ", g.residualRB,"\npairs:")
        print(g.pairs)
        input("mypairs")
        
        if gIdx == 0: # if this is the first group we need to substract the rate already given by the initialization
            consumedRBs = initializeRateForLowestRepresentation(groups)  # charge cost of lowest quality to lowest group, and allocate the lowest tiles to all groups.
            curRate = consumedRBs*groups[0].mcs
            generateTileRepresentationCostMatrixForLowestGroup(groups)
            #curUtil = g.Utility
        else:
            updateTileRepresentationCostMatrix(groups, gIdx) # update the cost matrix of each group based on the tile allocation of the previous group
            updateTileSelectionMatrix(groups, gIdx)          # update the selected tiles to those of the previous group
        
        while curRate <= g.maxRate:
            result = allocateBestTile(groups, gIdx)
            
            selectedTile     = result[0]
            selectedQuality  = result[1]
            selectedTileRBs  = result[2]
            selectedTileUtil = result[3]
            
            if selectedTile == -1 and selectedQuality == -1: # this means that all possible tiles have been allocated
                print("Allocated all best tiles for group idx ", gIdx)
                break;
            
            additionalRate = groups[gIdx].Ctm[ selectedTile ][ selectedQuality ]
            curRate += additionalRate
            #curUtil += selectedTileUtil
            
            g.Utility += selectedTileUtil
            g.pairs.append([g.RBs-g.residualRB, g.Utility])
            
            print("curRate ", curRate, "maxRate ", g.maxRate, "utility ", g.Utility)
            input("evaluate selected TILE")
            
            if curRate > g.maxRate:
                print("ALLOCATED RATE GREATER THAN MAX RATE\nGroup id", g.id)
                exit()
        
        g.curvefiting()
    




def getTotalGroupsUtility(groups):
    totalUtility = 0
    for g in groups:
        totalUtility += g.Utility*len(g.users)
        
    return totalUtility

def saveGroupState():
    print()

def crossLayer(competitors):

    global groupTiles, MaxSE, userGroups
    
    userGroups = []
    MaxSE = 0

    maxU = 0
    while True: # exit cross layer algorithm when new utility of the new grouping algorithm is not larger than the previous one
        
        print("--------- Iteration 1 ---------")
        # -- A
        crossLayergrouping()
        input("press enter")
        
        # -- B
        U = 0
        numGroups  = len(userGroups)
        perGroupRB = int(RBs/numGroups)
        groups = []
        for i in range(numGroups):
            groups.append(Group(i+1, perGroupRB, min(userGroups[i]), userGroups[i])) # create group class objects, i.e., the per group data
            
        #totalUtility = getTotalGroupsUtility(groups)
        
        round = 0
        while True: # to give the groups of the previous steps, exit when utility is not increased
            # -- B
            if round > 0:
                crossLayerResourceAllocation(groups)
                
            # -- C
            crossLayerRateSelection(groups)
        
            # save state at this point
            totalUtility = getTotalGroupsUtility(groups)
            
            print("PREV util", U, "\nNEW util", totalUtility,"\nnumGroups",len(groups),"\n", userGroups)
            input("EVALUATE")
            
            if totalUtility <= U:
                break;
                
            U = totalUtility
            round +=1
           
            if numGroups == 1:# no reason to make another round of resourse allocation when there is one group
                break;
           
        print("Exiting Iteration 2\n\tmaxU ", maxU,"\n\tU", U, "\n\tgroups ", numGroups)
        if maxU >= U:
            break
        
        print("maxU", maxU, "\tU",U)
        input("EXIting iteration2")
        maxU = U
        
    competitors[2].values.append(maxU)

def oneMulticastGroup(competitors):

    minMCS = min(user_mcs)

    problem = cplex.Cplex()
    problem.objective.set_sense(problem.objective.sense.maximize)
    problem.parameters.preprocessing.presolve.set(problem.parameters.preprocessing.presolve.values.off)
    #problem.parameters.timelimit.set(1000) # ~16 minutes
    problem.parameters.mip.tolerances.mipgap.set(optGap)

    # v.t.q
    for t in range(T):
        for q in range(Q):
            varName = "v." + str(t+1) + "." + str(q+1)
            objVal = math.log10(V[q])*tileWeights[t]
            problem.variables.add(obj=[objVal ],
                                    lb=[0.0],
                                    ub=[1.0],
                                    types=["B"],
                                    names=[varName])

    # sum_t( sum_q( v.t.q* b.t.q ) ) <= minMCS*RBs
    theVars=[]
    theCoefs=[]
    for t in range(T):
        for q in range(Q):
            varName = "v." + str(t+1) + "." + str(q+1)
            theVars.append(varName)
            theCoefs.append(V[q])
    
    problem.linear_constraints.add(lin_expr=[cplex.SparsePair(theVars, theCoefs)],
                                    senses=["L"],
                                    rhs=[minMCS*RBs])
    
    if debug == 1:
        print ("\t\t",theVars, theCoefs, " <= ", minMCS*RBs)
    

    # sum_q( v.t.q ) = 1
    for t in range(T):
        theVars=[]
        theCoefs=[]
        for q in range(Q):
            varName = "v."+str(t+1)+"."+str(q+1)
            theVars.append(varName)
            theCoefs.append(1)
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(theVars, theCoefs)],
                                        senses=["E"],
                                        rhs=[1.0])
        if debug == 1:
            print ("\t\t",theVars, theCoefs, " <= ", 1)
    
    
    problem.solve()
    printProblemSolution(problem)
    print("1-MG: Obj Value: ", problem.solution.get_objective_value()*I)
    RESULTS[0] += problem.solution.get_objective_value()*I
    competitors[0].values.append(problem.solution.get_objective_value()*I)
    
def optimalLTEVariables(problem):
    # y.k.t.q
    for k in range(K):
        for t in range(T):
            for q in range(Q):
                varName = "y." + str(k+1) + "." + str(t+1) + "." + str(q+1)
                problem.variables.add(obj=[0.0],
                                        lb=[0.0],
                                        ub=[1.0],
                                        types=["B"],
                                        names=[varName])
    # z.k
    for k in range(K):
        varName = "z." + str(k+1)
        problem.variables.add(obj=[0.0],
                              lb=[0.0],
                              ub=[ float(RBs) ],
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

                                  
    if debug ==1:
        print("\tNames: ", problem.variables.get_names())
        #print("\tTypes: ", problem.variables.get_types())
        #print("\t\nObj Function: ", problem.objective.get_linear())



def optimalLTEConstraints(problem):
   print("Generating problem constraints")


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
                                  
   
   # sum_t( sum_q( y.k.t.q*b.t.q ) ) <= z.k*c.k
   for k in range(K):
       theVars = []
       theCoefs = []
       for t in range(T):
           for q in range(Q):
               tmpName = "y." + str(k+1) + "." + str(t+1) + "." +str(q+1)
               theVars.append(tmpName)
               theCoefs.append(V[q])
               
       tmpName  = "z." + str(k+1)
       theVars.append(tmpName)
       theCoefs.append(-group_mcs[k])
       
       if debug == 1:
           print("\t\t", theVars, theCoefs, "<= 0")
       
       problem.linear_constraints.add(lin_expr=[cplex.SparsePair(theVars, theCoefs)],
                                       senses=["L"],
                                       rhs=[0.0])
   
   
   # sum_i( m.i.k.t.q ) - y.k.t.q <= 0
   for k in range(K):
       for t in range(T):
            for q in range(Q):
               theVars = []
               theCoefs = []
               for i in range(I):
                   tmpName = "m." + str(i+1) + "." + str(k+1) + "." + str(t+1) + "." +str(q+1)
                   theVars.append(tmpName)
                   theCoefs.append(1)
               
               tmpName = "y." + str(k+1) + "." + str(t+1) + "." + str(q+1)
               theVars.append(tmpName)
               theCoefs.append(-AUXILIARY)

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
           '''
           for w in range(AP):
               for q in range(Q):
                   tmpName = "x." + str(i+1) + "." + str(w+1) + "." + str(t+1) + "." + str(q+1)
                   theVars.append(tmpName)
                   theCoefs.append(1)
           '''
           
           if debug == 1:
               print("\t\t", theVars, theCoefs, "= 1")
               
           problem.linear_constraints.add(lin_expr=[cplex.SparsePair(theVars, theCoefs)],
                                           senses=["E"],
                                           rhs=[1.0])
   
   
   
   #sum_q( m.i.k.t.q*c.k ) <= c.i
   for i in range(I):
       for k in range(K):
           for t in range(T):
               theVars = []
               theCoefs = []
               for q in range(Q):
                   tmpName = "m." + str(i+1) + "." + str(k+1) + "." + str(t+1) + "." + str(q+1)
                   theVars.append(tmpName)
                   theCoefs.append(group_mcs[k])
                   
               if debug == 1:
                   print("\t\t", theVars, theCoefs, "<=", user_mcs[i])
                   
               problem.linear_constraints.add(lin_expr=[cplex.SparsePair(theVars, theCoefs)],
                                               senses=["L"],
                                               rhs=[user_mcs[i]])



def optimalLTE(competitors):

    problem = cplex.Cplex()
    problem.objective.set_sense(problem.objective.sense.maximize)
    problem.parameters.preprocessing.presolve.set(problem.parameters.preprocessing.presolve.values.off)
    #problem.parameters.timelimit.set(1000) # ~16 minutes
    problem.parameters.mip.tolerances.mipgap.set(optGap)
    
    
    optimalLTEVariables(problem)
    optimalLTEConstraints(problem)
    problem.solve()
    printProblemSolution(problem)
    print("O-LTE: Obj Value: ", problem.solution.get_objective_value())
    RESULTS[1] += problem.solution.get_objective_value()
    competitors[1].values.append(problem.solution.get_objective_value())


def optimalGWOLVariables(problem):

    print("\nGenerating GWOL problem variables")
    
    # y.k.t.q
    for k in range(K):
        for t in range(T):
            for q in range(Q):
                varName = "y." + str(k+1) + "." + str(t+1) + "." + str(q+1)
                problem.variables.add(obj=[0.0],
                                        lb=[0.0],
                                        ub=[1.0],
                                        types=["B"],
                                        names=[varName])
    
    
    # z.k
    for k in range(K):
        varName = "z." + str(k+1)
        problem.variables.add(obj=[0.0],
                              lb=[0.0],
                              ub=[ float(RBs) ],
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


def optimalGWOLConstraints(problem):

    print("Generating problem constraints")
    
    usersPerAp = numpy.zeros(AP).tolist()
    for i in range(I):
        for w in range(AP):
            if h_i_m[i][w] >0:
                usersPerAp[w] +=1
    #print(usersPerAp)
 
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
    
    '''
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
    '''
    
    
    # sum_t( sum_q( y.k.t.q*b.t.q ) ) <= z.k*c.k
    for k in range(K):
        theVars = []
        theCoefs = []
        for t in range(T):
            for q in range(Q):
                tmpName = "y." + str(k+1) + "." + str(t+1) + "." +str(q+1)
                theVars.append(tmpName)
                theCoefs.append(V[q])
                
        tmpName  = "z." + str(k+1)
        theVars.append(tmpName)
        theCoefs.append(-group_mcs[k])
        
        if debug == 1:
            print("\t\t", theVars, theCoefs, "<= 0")
        
        problem.linear_constraints.add(lin_expr=[cplex.SparsePair(theVars, theCoefs)],
                                        senses=["L"],
                                        rhs=[0.0])
    
    
    # sum_i( m.i.k.t.q ) - BigValue*y.k.t.q <= 0
    for k in range(K):
        for t in range(T):
             for q in range(Q):
                theVars = []
                theCoefs = []
                for i in range(I):
                    tmpName = "m." + str(i+1) + "." + str(k+1) + "." + str(t+1) + "." +str(q+1)
                    theVars.append(tmpName)
                    theCoefs.append(1)
                
                tmpName = "y." + str(k+1) + "." + str(t+1) + "." + str(q+1)
                theVars.append(tmpName)
                theCoefs.append(-AUXILIARY)

                if debug == 1:
                    print("\t\t", theVars, theCoefs, "<= 0")
                    
                problem.linear_constraints.add(lin_expr=[cplex.SparsePair(theVars, theCoefs)],
                                                senses=["L"],
                                                rhs=[0.0])
                    
    
    
    # sum_i( sum_w( x.t.w.t.q*b.t.q )) <= fixedWiFi Rate
    for i in range(I):
        for w in range(AP):
            theVars = []
            theCoefs = []
            for t in range(T):
                for q in range(Q):
                    tmpName = "x." + str(i+1) + "." + str(w+1) + "." + str(t+1) + "." + str(q+1)
                    theVars.append(tmpName)
                    theCoefs.append( V[q] )
            
            #tmpName = "a." + str(i+1) + "." + str(w+1)
            #theVars.append(tmpName)
            #theCoefs.append(-h_i_m[i][w])
            
            if debug == 1:
                print("\t\t", theVars, theCoefs, "<= 0")
                
            fixedWiFiRate = 0
            if h_i_m[i][w] >0:
                fixedWiFiRate = h_i_m[i][w]/usersPerAp[w]
                
            problem.linear_constraints.add(lin_expr=[cplex.SparsePair(theVars, theCoefs)],
                                            senses=["L"],
                                            rhs=[fixedWiFiRate])
                                            
       

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
    
    
    
    #sum_q( m.i.k.t.q*c.k ) <= c.i
    for i in range(I):
        for k in range(K):
            for t in range(T):
                theVars = []
                theCoefs = []
                for q in range(Q):
                    tmpName = "m." + str(i+1) + "." + str(k+1) + "." + str(t+1) + "." + str(q+1)
                    theVars.append(tmpName)
                    theCoefs.append(group_mcs[k])
                    
                if debug == 1:
                    print("\t\t", theVars, theCoefs, "<=", user_mcs[i])
                    
                problem.linear_constraints.add(lin_expr=[cplex.SparsePair(theVars, theCoefs)],
                                                senses=["L"],
                                                rhs=[user_mcs[i]])
    



def gwol(competitors):

    problem = cplex.Cplex()
    problem.objective.set_sense(problem.objective.sense.maximize)
    problem.parameters.preprocessing.presolve.set(problem.parameters.preprocessing.presolve.values.off)
    #problem.parameters.timelimit.set(1000) # ~16 minutes
    problem.parameters.mip.tolerances.mipgap.set(optGap)
    
    optimalGWOLVariables(problem)
    optimalGWOLConstraints(problem)
    problem.solve()
    printProblemSolution(problem)
    print("GWOL: Obj Value: ", problem.solution.get_objective_value())
    RESULTS[3]+=problem.solution.get_objective_value()
    competitors[3].values.append(problem.solution.get_objective_value())


def writeResults(iterations, numCompetitors):
    # initialize list of lists
    
    data = []
    names = ['1-MG', 'O-LTE', 'CROSS', 'GWOL', 'PUMA']
    
    for i in range(numCompetitors):
        curStd = -1
        if len(competitors[i].values) >0:
            curStd = statistics.stdev(competitors[i].values)
        data.append([names[i], RESULTS[i]/iterations, curStd])
    
    # Create the pandas DataFrame
    df = pd.DataFrame(data, columns = ['Name', 'Utility', 'Std'])
    
    print(Bm)
    fileName="RBs_"+str(RBs)+"_iters_"+str(iterations)+"_WiFi_"+str(Bm[0])+"_APs_"+str(AP)+".csv"
    
    df.to_csv(fileName)
    print(df)


class competitor:
    def __init__(self, name):
        self.name=name
        self.utility=0
        self.std=0
        self.groups=0
        self.values=[]
        
    def printInfo(self):
        print("Name: ", self.name,"\nUtility: ", self.utility, "\nstd: ", self.std,"\ngroups: ",self.groups,"\nValues: ", self.values)


if __name__ == "__main__":

    print("Panoramic Multicast Groups ( -- mrmo360 -- )")
    
    generateTileWeights(32)

    numCompetitors = 5 # 1-MG | O-LTE | CROSS | GWOL | PUMA
    RESULTS = numpy.zeros(numCompetitors) # 4 compe
    
    names = ['1-MG', 'O-LTE', 'CROSS', 'GWOL', 'PUMA']
    competitors = []
    for i in range(numCompetitors):
        competitors.append( competitor(names[i]) )
        #competitors[i].printInfo()
    
    readConfiguration()
    
    crossLayer(competitors)
    exit()
    
    iter =1
    maxIters = 2
    while iter <= maxIters:
        print("------------------------------------------------------------------------ITERATION ", iter)
        readConfiguration()
        
        oneMulticastGroup(competitors)
        optimalLTE(competitors)
        crossLayer(competitors)
        gwol(competitors)
        optimalPanoramicVideo(competitors)
        iter+=1
    
    writeResults(maxIters, numCompetitors)
