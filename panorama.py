import sys, json ,cplex, numpy, math
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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
debug = 1



# crossLayer Optimization : globecom_2018
MaxSE = 0
userGroups = []
groupResources = []
groupTiles = []

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
    
    

def optimalPanoramicVideo():

    problem = cplex.Cplex()
    problem.objective.set_sense(problem.objective.sense.maximize)
    problem.parameters.preprocessing.presolve.set(problem.parameters.preprocessing.presolve.values.off)
    #problem.parameters.timelimit.set(1000) # ~16 minutes
    
    problem.parameters.mip.tolerances.mipgap.set(0.05)
    
    generateProblemVariables(problem)
    generateProblemConstraints(problem)
    
    problem.solve()
   
    #printNewSolutionPerUser(problem)
    #printProblemSolution(problem)
    #printOutput(problem)
    
##############################################################################################################
##############################################################################################################


def getTileUtility(t,rate):
    return math.log10(rate)*tileWeights[t]

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
        
    
def calculateGroupResources():
    print(1)

def crossLayerResourceAllocation(groups):
    print("CrossLayer Resource Allocation")
    
    numGroups = len(groups)
    print("\tNumber of groups", numGroups)
    
    if numGroups == 1:
        groupResources.append(RBs)
        print("\tGroup RBs",  groupResources)
    else:
        calculateGroupResources()



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
                    effectiveUtil = getTileUtility(t, V[q])
                    tmp.append( effectiveUtil )
                else:
                    effectiveUtil = getTileUtility(t, V[q]) - getTileUtility(t, V[q-1])
                    tmp.append( effectiveUtil )
            self.effUtil.append(tmp)
    
    def curvefiting(self):
        print("curve fitting")
        x_data = []
        y_data = []
        
        for d in range(len(self.pairs)):
            x_data.append(self.pairs[d][0])
            y_data.append(self.pairs[d][1])
        
        x_data = numpy.array(x_data)
        y_data = numpy.array(y_data)
        
        print(x_data)
        print(y_data)
        print(self.pairs)
        popt, pcov = curve_fit(curveFitFunction, x_data, y_data, maxfev=100000)
        
        
        self.Ag = popt[0]
        self.Bg = popt[1]
        print(popt)
        
        #f= popt[0]*numpy.log10(x_data*popt[1])
        
        #fit = numpy.polyfit(x_data, y_data, 1)
        #f = fit[0]*x_data-fit[1]
        
        #plt.plot(x_data, y_data, "o")
        #plt.plot(x_data,f)
        #plt.show()
        
        #exit()
    
    def printGroupData(self):
        print("-------- Group Data --------\nId: ", self.id,"\nGroup Utility: ", self.Utility,"\neffUtil: ", self.effUtil, "\nUsers: ", self.users,"\nRBs: ", self.RBs,"\nResidual RB: ", self.residualRB ,"\nmcs: ", self.mcs, "\nrate: ", self.maxRate,"kbps\npais: ", self.pairs)
        print("Cost Matrix: ", self.Ctm)
        print("Tile Qualities: ", self.tiles)

        

def generateTileRepresentationCostMatrixForLowestGroup(groups):

    for t in range(T):
        tmpCost = []
        for q in range(Q):
            if q == 0:
                tmpCost.append(V[q])
            else:
                tmpCost.append(V[q] - V[q-1])
                
        groups[0].Ctm.append(tmpCost)
            
    '''
    else:
        for t in range(T):
            tmpCost = []
            for q in range(Q):
                prevIdx = idx - 1
                if q <= groups[prevIdx].tiles[t][q]: # if the previous groups has a same of better quality, then the cost is 0
                    tmpCost.append(0)
                elif q == groups[prevIdx].tiles[t][q] + 1: # if the previous groups has a lower tile quality, you allocate a new representaition so you pay all
                    tmpCost.append(V[q])
                elif q >= groups[prevIdx].tiles[t][q] + 2: # this step will come only after the previous elif, so you pay the difference from what you allocated in the prev elif
                    tmpCost.append(V[q] - tmpCost[q-1])
                    
            groups[idx].Ctm.append(tmpCost)
    '''


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
            g.tiles[t][0] = 0
            if g.id == 1: # allocate the necesary rate of the lower representation only to the first group
                totalRate += V[0]
                g.Utility += getTileUtility(t,V[0])

    consumedRB = math.ceil(totalRate/groups[0].mcs)
    print("INITIALIZING FIRST GROUP\n\tNecessary Rate: ", totalRate, "\tmcs: ", groups[0].mcs,"\tAvailable RB ", groups[0].RBs ,"\tRB needed ", consumedRB)
    
    groups[0].residualRB -= consumedRB
    groups[0].pairs.append([consumedRB, groups[0].Utility])
    groups[0].printGroupData()
    
    if consumedRB > groups[0].RBs:
        print("INFISIBLE RESOURCE ALLOCATION")
        exit()
        
    return consumedRB
    

def allocateBestTile(groups, idx):
        
    print("\n\nAllocating best tile for group idx", idx)
    
    foundBetterTile = False
    
    groups[idx].printGroupData()
    
    bestTileIdx = 0
    bestTileRB  = 0
    bestTileValue = 0
    bestTileQuality = 0
    for t in range(T):
        for q in range(Q):
            if groups[idx].tiles[t][q] == -1 and groups[idx].Ctm[t][q] != 0: # that means that this tile has not been allocated to the group yet
            
                #curValue =  getTileUtility(t,V[q])/groups[idx].Ctm[t][q]
                curValue = groups[idx].effUtil[t][q]/groups[idx].Ctm[t][q]
                neededRB = math.ceil(groups[idx].Ctm[t][q]/groups[idx].mcs)
                
                if curValue > bestTileValue and groups[idx].residualRB >= neededRB:
                    foundBetterTile = True
                    bestTileValue = curValue
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
        curUtil = 0
        curRate = 0
        
        if gIdx == 0: # if this is the first group we need to substract the rate already given by the initialization
            consumedRBs = initializeRateForLowestRepresentation(groups)  # charge cost of lowest quality to lowest group, and allocate the lowest tiles to all groups.
            curRate = consumedRBs*groups[0].mcs
            generateTileRepresentationCostMatrixForLowestGroup(groups)
            curUtil = g.Utility
        else:
            updateTileRepresentationCostMatrix(groups, gIdx) # update the cost matrix of each group based on the tile allocation of the previous group
            g.printGroupData()
            updateTileSelectionMatrix(groups, gIdx)          # update the selected tiles to those of the previous group
            g.printGroupData()
        
        
        input("Press Enter to continue...\n")
        
        
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
            #print("Previous Rate: ", curRate,"\t Additional Rate: ", additionalRate, "\tNew Rate: ", curRate+additionalRate)
            #print("Previous Util: ", curUtil, "\tNew Util: ", curUtil+math.log10(additionalRate)*tileWeights[ selectedTile ])
            curRate += additionalRate
            curUtil += selectedTileUtil
            
            g.Utility = curUtil
            g.pairs.append([g.RBs-g.residualRB, g.Utility])
            
            
            print("\n\nAFTER")
            g.printGroupData()
            input("Press Enter to continue...\n")
            
            if curRate > g.maxRate:
                print("ALLOCATED RATE GREATER THAN MAX RATE")
                exit()
        
        print("\n\n\n")
        g.printGroupData()
        g.curvefiting()
        exit()


def getTotalGroupsUtility(groups):
    totalUtility = 0
    for g in groups:
        totalUtility += g.Utility
        
    return totalUtility

def crossLayer():

    global groupTiles

    U = 0
    maxU = -1
    while maxU < U:
    
        maxU = U
        
        # -- A
        for i in range(2):
            crossLayergrouping()
        input("press enter\n")
        # -- B
        U = 0
        numGroups  = len(userGroups)
        perGroupRB = int(RBs/numGroups)
        groups = []
        for i in range(numGroups):
            groups.append(Group(i+1, perGroupRB, min(userGroups[i]), userGroups[i])) # create group class objects, i.e., the per group data
 
        # -- B
        crossLayerResourceAllocation(groups)
 
        # -- C
        crossLayerRateSelection(groups)
 
        '''
        while 1:
            calculateResourceAllocationAndRateSelection(groups)
            if U < getTotalGroupsUtility(groups):
                U = getTotalGroupsUtility(groups)
            else:
                break;
        '''
    


if __name__ == "__main__":

    print("Panoramic Multicast Groups ( -- mrmo360 -- )")
    
    generateTileWeights(32)
    readConfiguration()
    
    crossLayer()
    
    optimalPanoramicVideo()
    
