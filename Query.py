import random

import numpy as np

import loadDataSet

from loadDataSet import countBinaryDig
def interlaceTokenGen(D,seed=None,Qtable=None,defaultZ= False):
    maxCol = np.max(D, 0)
    BinaryDig = []
    for maxci in maxCol:
        BinaryDig.append(loadDataSet.countBinaryDig(maxci))
    r, c = D.shape
    if seed == None:
        np.random.seed(42)
    else:
        np.random.seed(seed)
    tolbit =0
    for v in BinaryDig:
        tolbit+=v
        if v == 0:
            tolbit+=1
    colStack = {}
    for i in range(c):
        colStack[i] = []
    defaultZL= []
    for loopIter in range(max(BinaryDig)):
        cordinate = []
        for columnIdx,lli in enumerate(BinaryDig):
            if lli == 0:
                lli=1
            if loopIter >= lli:
                continue
            else:
                colStack[columnIdx] .append(str(columnIdx)+'-'+str(loopIter))
                cordinate.append(  str(columnIdx)+'-'+str(loopIter))
                if defaultZ == True:
                    defaultZL.append( str(columnIdx)+'-'+str(loopIter))
    if defaultZ:
        return defaultZL
    for i in colStack.keys():
        colStack[i].reverse()
    if Qtable == None:
        Qtable = np.zeros((tolbit,c))+ 0.2
    token = []
    for i in range(tolbit):
        #check Prob
        for j in range(c):
            if len(colStack[j])==0:
                Qtable[i][j] = 0
        Qtable[i]/=(sum(Qtable[i]))
        sam = np.random.multinomial(  1,Qtable[i],1)[0]
        for j in range(sam.shape[0]):
            if sam[j] == 1:
                v = colStack[j].pop()
                token.append(v)
    return  token


class Query:
    def __init__(self,center,width):
        lu , rd = [],[]
        self.leftup = (center - width/2)
        c, = self.leftup.shape
        for i in range(c):
            if self.leftup[i] < 0:
                self.leftup[i] = 0
            self.leftup[i] = int(self.leftup[i])
            lu.append(int(self.leftup[i]))

        self.rightdown = center+ width/2
        for i in range(c):
            self.rightdown[i] = int(self.rightdown[i])
            rd.append(int(self.rightdown[i]))
        self.leftup=lu
        self.rightdown = rd
        self.trueNumber = 0
        self.BinaryDig = None
    def assignBinaryDig(self,BD):
        self.BinaryDig = BD
    def assignTN(self,tn):
        self.trueNumber = tn
    def inQBox(self,tuple):
        c = len(self.leftup)
        for i in range(c):
            if tuple[i] >= self.leftup[i] and tuple[i]<= self.rightdown[i]:
                continue
            else:
                return False
        return True

    def __str__(self):
        s0 = ""
        for v in self.leftup:
            s0+= str(int(v))
            s0+="\t"
        s0+="\n"
        for v in self.rightdown:
            s0 += str(int(v))
            s0 += "\t"
        return  s0

    def KDBS(self):
        s0 = ""
        for v in self.leftup:
            s0 += str(int(v))
            s0 += "\t"
        for v in self.rightdown:
            s0 += str(int(v))
            s0 += "\t"
        return s0

def scan(D,Query,dem=False):
    r,c = D.shape
    cnt = 0
    # print(r,c)
    for i in range(r):
        tuple = D[i,:]
        if Query.inQBox(tuple):
            cnt+=1
            if dem ==True:
                print("Rid:",i,tuple)
    return cnt
def scanfast(D,Query,dem=False):
    r,c = D.shape
    cnt = 0
    # print(r,c)
    ind = np.ones((r, )).astype(bool)
    for j in range(c):
        lb = Query.leftup[j]
        ub = Query.rightdown[j]
        # print(lb,ub)
        # val =  ( D[:,j] >= lb)
        # print(val.shape)
        ind &= ( D[:,j] >= lb)
        ind &= (D[:, j] <= ub)
    return np.sum(ind,axis=0)
    #
    # for i in range(r):
    #     tuple = D[i,:]
    #     if Query.inQBox(tuple):
    #         cnt+=1
    #         if dem ==True:
    #             print("Rid:",i,tuple)
    # return cnt

def QueryGen(D,QueryNumber,sel=1.0):
    np.random.seed(1410)
    r,c = D.shape
    print(r,c)
    tok = interlaceTokenGen(D, defaultZ=True)
    maxCol = np.max(D, 0)
    BinaryDig = []
    for maxci in maxCol:
        BinaryDig.append(countBinaryDig(maxci))
    # miniLen = 20000
    # D = D[:miniLen,:]
    # tuplesIdx = np.random.randint(0,miniLen,size=(QueryNumber))
    tuplesIdx = np.random.randint(0,r,size=(QueryNumber*100))

    SelectRows = D[tuplesIdx,:]

    domain = np.max(D,0) - np.min(D,0)
    # print(domain)
    # print(np.max(D,0))
    Querys = []
    # cet = [ 7   , 1980     ,  0   ,80815   ,    1     , 66]
    # wi = []
    # # for j in range(c):
    # #     wi.append(9999999)
    # wi[0] = 0
    # Q =Query(center=np.array(cet),width=np.array(wi))
    # cnt = 0
    # for i in range(r):
    #     if D[i,0] == 7:
    #         cnt+=1
    # print(cnt)
    # print(Q)

    # exit(1)
    # for j in range(c):
    #     wi.append()
    allCard=0
    QH = []
    QL = []
    QExL = []
    QX = []
    QHcard=0
    QLcard=0
    QExLcard=0
    presel=1
    maxCN = 2
    minCN = 1
    # presel=0
    for i in range(QueryNumber*10):
        print(len(QH),len(QL),len(QExL),len(QX))
        print('generating Q:',i)
        if(len(QH)>=QueryNumber):
            presel = 0.01
            maxCN = int(c/2)+1
        if(len(QL)>=QueryNumber):
            presel = 0.0001
            maxCN = c
        if (len(QExL) >= QueryNumber):
            presel = 0.0
            maxCN  = c
        wi = []
        selcols = random.randint(minCN,maxCN)
        idxs = np.random.choice(c,replace =False,size=selcols)
        for j in range(c):
            if j in idxs:
                wi.append(presel*random.randint(0,domain[j]+1 ))
            else:
                wi.append(domain[j]*2)
        Q= Query(center=SelectRows[i,:],width=np.array(wi))
        # if i==0:
        #     termN = (scan(D,Q,True))
        #     print("--------")
        #     print("Q:")
        #     print(Q)
        # else:
        # print(D.shape)
        # termN= scan(D[:200000 ,:] ,Q,False)
        card = scanfast(D[: ,:],Q,False)
        cardsel = card/r
        allCard+=(card/r)
        Q.assignTN(card)
        Q.assignBinaryDig(BinaryDig)
        # Querys.append(Q)

        # if card <=2:
        #     QX.append(Q)    
        # if len(QX) >=querynum:
        #     break
        # continue
        if cardsel >=0.01:
            if (len(QH) < querynum):
                QH.append(Q)
                QHcard+=card
        elif cardsel >=0.0001:
            if (len(QL) < querynum):
                QL.append(Q)
                QLcard+=card
        else:
            if (len(QExL) < querynum):
                QExL.append(Q)
                QExLcard+=card
            else:
                if card == 1:
                    QX.append(Q)

        if len(QH)>=querynum and len(QL)>=querynum and len(QExL)>=querynum  :
            break
        print( card)

        # BinaryrowList = []
        # for j in range(c):
        #     if BinaryDig[j] == 0:
        #         BinaryrowList.append([0])
        #         continue
        #     BinaryrowList.append(loadDataSet.number2Binary(SelectRows[i,j], BinaryDig[j]))
        # print(BinaryrowList)
        # print(loadDataSet.binaryRow2Zencode(BinaryrowList ))
        # bl = loadDataSet.binaryRow2TokZencode(BinaryrowList,tok)
        # for v in bl:
        #     print(v,end='')
        # print()
        # print(len(bl))
        # print(Q)
        # exit(1)
        # if termN > 100000:
        #     continue
        # else:
        
    # wi = []
    #
    # for Q in Querys:
    #     Q.assignTN(scan(D,Q))
    #     Q.assignBinaryDig(BinaryDig)
    print('AvgCardSel:',allCard/(querynum))
    print(QHcard/(r*100),QLcard/(r*100),QExLcard/(r*100))
    print(len(QH),len(QL),len(QExL),len(QX))
    return QH,QL,QExL,QX
    # exit(1)
    return Querys

def writeQs(Qs,filePath= './data/IMDBQuerys.txt'):
    s0 = ""

    s0+=str(len(Qs[0].BinaryDig))
    s0+='\t'
    s0+=str(len(Qs))
    s0+='\n'
    for v in Qs[0].BinaryDig:
        s0+=str(v)
        s0+='\t'
    s0+='\n'

    for q0 in Qs:
        s0+=str(q0)
        s0+='\n'
        s0+=str(q0.trueNumber)
        s0+='\n'
    f = open(filePath, mode='w')
    f.write(s0)

def KDBStyleQGen(Qs,filePath):
    s0 = ""
    # s0+=str(len(Qs[0].BinaryDig))
    # s0+='\t'
    # s0+=str(len(Qs))
    # s0+='\n'
    # for v in Qs[0].BinaryDig:
    #     s0+=str(v)
    #     s0+='\t'
    # s0+='\n'
    for q0 in Qs:
        s0+='r '
        s0+=(q0.KDBS())
        s0+='\t'
        s0+=str(q0.trueNumber)
        s0+='\n'
    f = open(filePath, mode='w')
    f.write(s0)

def regenerateCSV():
    D = np.load('./data/powerOri.npy')
    r,c = D.shape
    f = open(r'./contrastExp/CardBaseline/power.csv', mode='w')
    for j in range(c):
        f.write('col'+str(j)+';')
    f.write('\n')
    for i in range(r):
        for j in range(c):
            f.write(str(int(D[i,j]))+';')
        f.write('\n')
    f.close()
import sys
import pandas as pd
if __name__ == "__main__":
    querynum = int(sys.argv[1])
    sel = float(sys.argv[2])
    data = sys.argv[3]
    outfile = sys.argv[4]
    # outfilename = sys.argv[4]
    print('querynum get:',querynum)
    print('avg select ratio: ',sel)
    print('dataget ',data)
    print('FileOut:',outfile)
    # exit(1)
    if data == 'osm':
        decfile = './data/osmfile.npy'
        D = np.load(decfile)
        QH,QL,QExL,QX =  QueryGen(D,querynum,sel)
        writeQs(QH,outfile )
        writeQs(QL,outfile+'001')
        writeQs(QExL,outfile+'00001')
    elif data =='power':
        decfile = './data/powerOri.npy'
        D = np.load(decfile)
        QH,QL,QExL,QX =  QueryGen(D,querynum,sel)
        writeQs(QH,outfile+'1')
        writeQs(QL,outfile+'001')
        writeQs(QExL,outfile+'00001')
 
    elif data == 'DMV':
        decfile = './data/DMVint.npy'
        D = np.load(decfile)
        QH,QL,QExL,QX =  QueryGen(D,querynum,sel)
        writeQs(QH,outfile+'1')
        writeQs(QL,outfile+'001')
        writeQs(QExL,outfile+'00001')
 
    
