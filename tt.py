from numpy import *
from scipy.optimize import *


def myFunction(z):
    x = z[0]
    y = z[1]
    w = z[2]

    F=empty((3))
    F[0] = pow(x,2)+ pow(y,2)-20
    F[1] = y-pow(x,2)
    F[2] = w+5 -x*y
    return F


if __name__ == "__main__":

    zGuess=array([1,1,1])
    z=fsolve(myFunction, zGuess)
    print(z)
