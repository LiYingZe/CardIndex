import functools
import random
import time
import torch
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
import numpy as np

import loadDataSet
import sys
from CITree import tupleList2Page
from CIndex import tuplez, cmp2List, cmp2Listbel
import sys

class MaskedLinear(nn.Linear):
    """ 带有mask的全连接层 """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))# 必须提前注册buffer才能够使用

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)

class MyMADE(nn.Module):
    def __init__(self,Xlength=6,innnerDepth = 30,linearScaleN = 5):
        super(MyMADE, self).__init__()
        innnerDepth=Xlength
        # mk : order
        MK= []
        m0 = []
        m3 = []
        for i in range(Xlength):
            m0.append(i)
            m3.append(i)
        m1 = []
        # m2 = []
        for i in range(innnerDepth):
            m1.append( i)
            # m2.append( random.randint(1,Xlength-1))
        MK.append(m0)
        MK.append(m1)
        # MK.append(m2)
        MK.append(m3)
        # print(m0)
        # print(m1)
        # print(m3)
        self.maskList = []
        iolengthList = [ [ Xlength,innnerDepth], [innnerDepth,Xlength]]
        idx = 0
        for L in iolengthList:
            idx+=1
            i0 = L[0]
            j0 = L[1]
            mask = np.zeros((i0,j0))
            # print(i0,j0)
            for i in range(i0):
                for j in range(j0):
                    maskp0 = MK[idx-1]
                    maskp1 = MK[idx]
                    # print(i,j, maskp0[i], maskp1[j])
                    if maskp0[i] < maskp1[j] and  maskp0[i] >= (maskp1[j]-linearScaleN):
                        mask[i][j] = 1
                    else:
                        mask[i][j] = 0
            self.maskList.append(mask)
        self.fc1 = MaskedLinear(Xlength , innnerDepth)#0-1
        self.fc1.set_mask(self.maskList[0])
        # self.fc2 = MaskedLinear(innnerDepth,innnerDepth)#1-2
        # self.fc2.set_mask(self.maskList[1])
        self.fc3 = MaskedLinear(innnerDepth,Xlength)#2-3
        self.fc3.set_mask(self.maskList[1])
        # self.cachex = None
        # self.cachex2 = None
        # self.cachex3 = None
    def forward(self,x):
        a,b = x.shape
        x = x.view(a,-1).float()
        # print(x.shape)
        x = F.relu(self.fc1(x))
        # self.cachex = x
        # x = F.relu(self.fc2(x))
        # self.cachex2 = x
        x = torch.sigmoid(self.fc3(x))
        # print(x.shape)
        return x


class MyMLP(nn.Module):
    def __init__(self):
        """
        把MADE的输出迁移到桶的输出中
        """
        super(MyMLP,self).__init__()
        self.linear1 = nn.Linear(6,12)
        self.linear2 = nn.Linear(12,6)

    def forward(self,x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = torch.sigmoid(x)
        return x

def ExploreR(D):
    D = np.array(D)
    r, c = D.shape
    print(r,c)
    for i in range(c):
        print("col ",i," ndv:",len(set(D[:,i])))
def testIndexCap():
    ZD = np.load('./data/power-ZD.npy')
    r,c = ZD.shape
    tl=[]
    net = torch.load("./Model/MADE.pt").cuda()
    for i in range(r):
        zxi = ZD[i,:]
        tuplezx = tuplez(None, zxi,0)
        tl.append(tuplezx)
    tl = sorted(tl, key=functools.cmp_to_key(cmp2List))
    print("sorted")
    cdfDistance = []
    for i in range(r):
        if i % 1000==0:
            out = net(torch.tensor(tl[i].z).cuda().view(1,-1))
            acc=1
            cdf=0
            for j in range(c-1):
                oneProb = out[0,j]
                v = tl[i].z[j]
                cdf += (acc * (1 - oneProb) * v)
                acc *= (oneProb * v + (1 - oneProb) * (1 - v))
            print("Real cdf:", i / r,"est",cdf)
            cdfDistance.append(abs((i/r)-cdf))
    print('maxdis:',max(cdfDistance),'mindis',min(cdfDistance),'avgdis',sum(cdfDistance)/len(cdfDistance))

def trainMADE(ZD,traiingIter = 10,dataname=None,link=30):
    # ZD = np.load(zdpath)
    # net = torch.load('./Model/MADE.pt')
    r, c = ZD.shape
    # ZD = torch.tensor(ZD).cuda()
    print(r,c)
    net = MyMADE(c, c,link).cuda()
    ZD = loadDataSet.datainBinary(ZD)
    batchS = 1024
    dataiter = DataLoader(ZD, batch_size=batchS, shuffle=True)
    losfunc = nn.BCELoss()
    optimizer = optim.Adam(net.parameters())
    batch_idx = 0
    for iii in range(traiingIter):
        for (x, target) in (dataiter):
            # r, c = x.shape
            # xnew = torch.cat([torch.zeros((r, 1)).cuda(), x[:, :-1].cuda()], dim=1).cuda()
            optimizer.zero_grad()
            out = net(x.cuda() + 0.0)
            # print(out)
            loss = losfunc(out, x.cuda() + 0.0)
            loss.backward()
            optimizer.step()
            batch_idx += 1
            if batch_idx % 100 == 0:
                print('\r  ',batch_idx,'/', (r*traiingIter)/batchS,' loss',float(loss),end=' ',flush=True)
                # print(out[0,:])
                # print()
                # print(xnew[0,:])
                # print()
                # print(x[0,:])
                # exit(1)
                torch.save(net, './Model/MADE'+dataname+'.pt')
    torch.save(net, './Model/MADE'+dataname+'.pt')


def MADE2File(madePath,outFileName,zdr,zdc,connectlen,leafnum):
    net = torch.load(madePath)
    parameterL = []
    swrite=""
    swrite+=(str(zdr)+" "+str(zdc)+" "+str(connectlen)+" "+str(leafnum))

    swrite+="\n"
    idx = 0
    for name, p in net.named_parameters():
        print(name)
        print(p.shape)
        pa = p.cpu().detach().numpy()
        if 1 in pa.shape:
            pa = pa.reshape(-1)
        parameterL.append(pa.tolist())
        idx += 1
    s0 = ""
    for it in parameterL:
        sit = str(it)
        sit = sit.replace('[', ' ')
        sit = sit.replace(']', '\n')
        sit = sit.replace(',', ' ')
        s0 += sit
        s0 += '\n'
    swrite+=s0
    f = open(outFileName, mode='w')

    f.write(swrite)
    f.close()
def rootPartition(D,ZD,subNum=100,govbit = 20,dataname=None):
    leafNodeTuple = []
    batch_size = 1000
    govBit = govbit
    for i in range(subNum):
        leafNodeTuple.append([])
    minnimumsep = 1.0 / subNum
    r, c = ZD.shape
    r00=r
    rootFile = './Model/MADE'+dataname+'.pt'
    net = torch.load(rootFile)
    ZD = torch.tensor(ZD.copy())
    dataSet = loadDataSet.datainBinary(ZD)
    dataiter = DataLoader(dataSet, batch_size=batch_size, shuffle=False)
    idx = 0
    rx = r
    innnercnt = 0
    # net.eval()
    belongValueList = torch.zeros((r,1)).int()
    startidx=0
    t0 = time.time()
    binaryz64 = torch.zeros((r,1)).long()
    for  i in range(min(64,c)):
        binaryz64 *=2
        binaryz64 += ZD[:,i].view(-1,1)
    binaryz64 = binaryz64.cpu().numpy()
    tmid  = time.time()
    print('zencode calculated',tmid-t0)

    with torch.no_grad():
        for (x, target) in (dataiter):
            print('\r',idx, '/', rx / batch_size, end=' ',flush=True)
            # print(x.shape,x.dtype,sys.getsizeof(x))
            # print(target.shape,target.dtype,sys.getsizeof(target))
            idx += 1
            # t0 = time.time()
            x = x.cuda()
            out = net(x)
            r, c = out.shape
            # 并行化替代
            acc = torch.ones((r, 1)).cuda()
            cdf = torch.zeros((r, 1)).cuda()
            for j in range(govBit - 1):
                v = x[:,j].view(-1, 1).cuda()
                oneProb = out[:, j].view(-1, 1).cuda()
                cdf += (acc * (1 - oneProb) * v)
                acc *= (oneProb * v + (1 - oneProb) * ( ~ v))

            t1 = time.time()
            x = x.cpu().detach().numpy()
            # print(x[0])
            # exit(1)
            # tx = time.time()
            belongl = (cdf/minnimumsep).int()
            # print(belongl[:10,0])
            # print(belongl[:10, 0].int())
            belongValueList[startidx:startidx+r,:] = belongl
            startidx+=r
            # tuplezx = tuplez((D[innnercnt, :]), x[0,:], cdf[i, 0])
            #
            # for i in range(r):
            #     belong = belongl[i,0]
            #     xi = x[i,:]
            #     tuplezx = tuplez((D[innnercnt, :]), xi, cdf[i, 0])
            #     # print(sys.getsizeof(tuplezx))
            #     # exit(1)
            #     # leafNodeTuple[belong].append(tuplezx)
            #     innnercnt += 1
            t2 = time.time()
            # break
            # print(t1 - t0, t2 - t1, tx - t1)
    print(belongValueList.shape)
    t1 = time.time()
    print(t1-t0)
    tups = np.zeros((r00,2)).astype('int64')
    # tups = []
    for i in range(r00):
        if i%10000==0:
            print('\r',i,'/',r00,end=' ',flush=True)
        tups[i,1] = belongValueList[i,0]
        tups[i,0] = binaryz64[i,0]
        # tups[i,0 ] = i

        # tups.append(tuplez(i,i,belongValueList[i,0],binaryenco32[i,0]))
    # tups = [ tuplez(i,i,belongValueList[i,0],binaryenco32[i,0]) for i in range(r00)]
    # print(tups)
    t2 = time.time()
    print("TL created",t2-t1)
    print('sorting:')
    #debuging
    # tups*=0
    ind = np.lexsort(tups.T)

    # tups = sorted(tups,key=functools.cmp_to_key(cmp2Listbel))
    t3 = time.time()
    # print(tups[ind])
    # exit(1)
    print('sortdone , takes: ',t3-t2)
    # exit(1)
    # print(len(tups))
    # for i in range(10):
    #     print(tups[i].belong)
    r = len(tups)
    curentleafidx =  0
    lasti=0
    subpage = []
    for vi in range(r):
        indi = ind[vi]
        tupbelong = tups[indi,1]
        if tupbelong > curentleafidx:
            print("switching leaf node to ",tupbelong,'old : ',curentleafidx, "leafrecords: ",vi-lasti )
            if (vi- lasti )== 0:
                subpage.append(tuplez(D[indi, :], ZD[indi, :], tups[indi, 1], binaryz64[indi,0]))
                continue
            # print(len(subpage))
            tupleList2Page("N-"+str(curentleafidx), subpage,leafSubName=dataname)
            curentleafidx = int(tupbelong)
            lasti=vi
            subpage=[]
            subpage.append(tuplez(D[indi, :], ZD[indi, :], tups[indi, 1], binaryz64[indi,0]))
        else:
            subpage.append( tuplez(D[  indi, :],ZD[indi,:], tups[indi,1],binaryz64[indi,0]) )
            # print(len(subpage))
    if lasti !=(r-1):
        print("fixing the last one:")
        tupleList2Page("N-" + str(int(curentleafidx)), subpage,leafSubName=dataname)


    print('done')
    return
    # exit(1)
    # for i in range(len(leafNodeTuple)):
    #     if len(leafNodeTuple[i])==0:
    #         continue
    #     print("sorting: leaf",i,"leaf len:",len(leafNodeTuple[i]))
    #     leafNodeTuple[i] = sorted(leafNodeTuple[i],key=functools.cmp_to_key(cmp2List))
    #     print("giving control 2 linear M")
    #     tupleList2Page("N-"+str(i),leafNodeTuple[i])
    #     leafNodeTuple[i]=[]

def trainprocedure(decfile,zfile,trainiter,leafnum,dataname=None,link=30):
    D = np.load(decfile).astype(int)
    ZD = np.load(zfile).astype(bool)
    # print(ZD.shape)
    # print(D.shape)
    # print(D[0,:])
    # print(ZD[0,:])
    # endxit(1)
    trainStart = time.time()
    if trainiter!=0:
        trainMADE(ZD,trainiter,dataname=dataname,link=link)
    # net = torch.load('./Model/MADE.pt')
    trainend = time.time()

    rootPartition(D, ZD, leafnum, 30,dataname)
    r,c = ZD.shape
    leafnum = leafnum
    connectlen = link
    MADE2File('./Model/MADE'+dataname+'.pt', './Model/MadeRoot'+dataname, r, c, connectlen, leafnum)
    return trainend-trainStart
    # MADE2File('./Model/MADE.pt', './Model/MadeRoot.txt')

if __name__ =="__main__":
    leafs = int(sys.argv[1])
    trainloops = int(sys.argv[2])
    data = sys.argv[3]
    print('leafsNumber: ',leafs,'Training loops ',trainloops,'data: ',data)
    t0 = time.time()
    if data == 'osm':
        decfile = './data/osmfile.npy'
        zfile = './data/osmZD.npy'
        trainT=trainprocedure(decfile, zfile, trainloops, leafnum=leafs,dataname=data)

    elif data == 'power':
        decfile = './data/powerOri.npy'
        zfile = './data/power-ZD.npy'
        trainT=trainprocedure(decfile, zfile, trainloops, leafnum=leafs,dataname=data)

    elif data == 'DMV':
        decfile = './data/DMVint.npy'
        zfile = './data/DMV-ZD.npy'
        trainT=trainprocedure(decfile, zfile, trainloops, leafnum=leafs,dataname=data)
    t1 = time.time()
    print('construction takes:',t1-t0)
    print("Traing Takes:" ,trainT)
    f = open('./result/Construct'+data,'w')
    f.write('Building Time:'+str(t1-t0))
    f.write("\nTraining:"+str(trainT))
    f.close()
    