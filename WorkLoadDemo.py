import numpy as np
def loadQuerysSelGen(qfnRoot,rowsN,ofn):
    of = open(ofn,'w')
    of.write('1 ,001 ,00001 \n')
    # QuerySel = np.zeros(())
    Qsel = []
    for postname in [ '1','001','00001']:
        qf  = open(qfnRoot+postname)
        head = qf.readline()
        useless  = qf.readline()
        c,r = head.split('\t')
        r = int(r)
        c = int(c)
        cardList = []
        for i in range(r):
            rowiup = qf.readline()
            rowidown = qf.readline()
            card = qf.readline()
            cardList.append(int(card))
        # print(cardList)
        cardList = np.array(cardList)/rowsN
        # print(cardList)
        Qsel.append(cardList)
    Qsel = np.array(Qsel).T
    r,c = Qsel.shape
    for i in range(r):
        for j in range(c):
            of.write(str(float(Qsel[i,j])))
            of.write(' , ')
        of.write('\n')
    of.close()
    print(Qsel.shape) 

if __name__ =="__main__":
    loadQuerysSelGen('./data/DMV',12300116,'./result-merged/DMVSel.csv')
