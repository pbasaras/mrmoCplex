from numpy import *
from scipy.optimize import *
from sympy.interactive import printing
printing.init_printing(use_latex=True)


def myFunction(z):
    x = z[0]
    y = z[1]
    w = z[2]

    F=empty((3))
    F[0] = pow(x,2)+ pow(y,2)-20
    F[1] = y-pow(x,2)
    F[2] = w+5 -x*y
    return F
    
    
def newFunction(z):

    F=empty((3))
    F[0] = z[0]*z[1] + 4
    F[1] = z[0]*z[2] - 8
    return F
    
    
def test(z):

    F=empty((2))
    F[0] = 2*z[0]+z[1] -4
    F[1] = z[0] +z[1] -8
    return F

    
def mySum(z):
    
    F=empty((2))
    F[0]= sum(z)-5
    F[1] = z[0]+4*z[1]-12
    return F




def myArgs(z):
    
    F = empty((2))
    F[0] = -4
    for i in range(2):
        F[0] = F[0]+ z[i]
        
    F[1] = 2*z[0]+4*z[1] -28

    return F

if __name__ == "__main__":


    
    t= fsolve(myArgs,[1,1])
    print(t)



    exit()

    zGuess=array([])
    z=fsolve(myFunction, zGuess)
    print(z)
    
    
    print("\n------\n")
    
    
    p=fsolve(newFunction,[1,1,1])
    print(p)
    
    
    
    exit()
    print("\n------\n")
    g=fsolve(test,[1,1])
    print(g)
    
    
