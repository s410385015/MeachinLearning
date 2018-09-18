import numpy as np 
import matrix

"""
Read data points from the file,
and push it into 2d list array as float type
"""
def ReadFile(path):
    file=open(path,'r')
    tmp=file.read().split('\n')
    data=[]
    for i in range (0,len(tmp)):
        data.append(tmp[i].split(','))
    
    data=np.array(data)
    data=data.astype(np.float)
    
    return data


"""
For the data points set,generate the design matix A and the target matrix B in Ax-b
eg.
    data point (1,2) (3,4) (5,6) (7,8) base= 3

design matrix A:                        target matrix B:
    | x0^2 x0 1 |     |  1  1  1 |          | y0 |      | 2 |  
    | X1^2 X1 1 |  =  |  9  3  1 |          | y1 |   =  | 4 |          
    | X2^2 X2 1 |     | 25  5  1 |          | y2 |      | 6 | 
    | X3^2 X3 1 |     | 49  7  1 |          | y3 |      | 8 |
"""
def FindAandB(data,base):
    A=[]
    for i in range(0,len(data)):
        tmp=[data[i][0]**j for j in range(base-1,-1,-1)]
        A.append(tmp)
    A=np.array(A)

    B=[[data[i][1]] for i in range(0,len(data))]
    B=np.array(B)

    return A,B

def printEquation(_x):
    eq=""
    for i in range (len(_x)-1):
        eq+="{:.2}".format(_x[i][0])+"x^"+str(len(_x)-i-1)+"+"
    eq+="{:.2}".format(_x[len(_x)-1][0])
    eq+="=y"
    print(eq)


def main():

   
    path=input("Enter file name with path(eg. D:\lecture1\data.txt):")
    base=input("Enter the number of polynomial bases:")
    l=input("Enter lambda:")

    base=int(base)
    l=float(l)
    #d=ReadFile('D:\桌面用\MeachinLearning\MLclass\lecture1\data.txt')
    #d=ReadFile('data.txt')
    #base=3
    #l=0.5
    d=ReadFile(path)
    A,B=FindAandB(d,3)
    matrix.printMatrix(A,"A")
    matrix.printMatrix(B,"B")


    AT=matrix.TransposeMatrix(A)
    matrix.printMatrix(AT,"AT")

    ATA=matrix.Mul(AT,A)
    matrix.printMatrix(ATA,"ATA")

    eye=matrix.eyeMatrix(3,0.5)
    _ATA=matrix.Add(ATA,eye)
    matrix.printMatrix(_ATA,"ATA+λi")

    L,U=matrix.FindLU(_ATA)
    matrix.printMatrix(L,"L")
    matrix.printMatrix(U,"U")

    ATAi=matrix.FindInverseFromLU(L,U)
    matrix.printMatrix(ATAi,"ATAi")

    
    _x=matrix.Mul(matrix.Mul(ATAi,AT),B)
    matrix.printMatrix(_x,"_x")
    printEquation(_x)

    error=matrix.Sub(matrix.Mul(A,_x),B)
    
    w_error=matrix.Mul(matrix.TransposeMatrix(_x),_x)
    w_error=matrix.Factor(w_error,l)
    matrix.printMatrix(w_error,"weight error")
    value_error=matrix.Mul(matrix.TransposeMatrix(error),error)
    matrix.printMatrix(value_error,"value error")
    matrix.printMatrix(matrix.Add(value_error,w_error),"error")
    
    
        
if __name__ =="__main__":
    main()