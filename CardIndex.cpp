
#include <iostream>
#include <math.h>
#include <chrono>
#include <fstream>
#include <string>
#include <random>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
using namespace std;
using namespace chrono;

typedef struct MADENet
{
    int zdr;
    int zdc;
    int connectlen;
    int leafnums = 100;
    int diglen;
    float *fc1w;
    float *fc1b;
    float *fc2w;
    float *fc2b;
} MADENet;
typedef struct Query
{
    int queryid;      // 唯一编码Queryid
    int columnNumber; // how many columns
    int *binaryLength;
    long long *leftupBound;
    long long *rightdownBound;
} Query;
typedef struct zpage
{
    int r;
    int c;
    int digs;
    bool minz[128];
    bool maxz[128];
    long long *data;
} zpage;
typedef struct LinearRegression
{
    double *kup;
    double *kdown;
    int lens;
    long long *keys;
    long *rows;
    // float intercept;
    // int xlen;
    // int subN;
} LinearRegression;
typedef struct BNode
{
    int itemN;
    int Zvalues[400][128];
    int childNumber[400];
} BNode;

typedef struct CITLeaf
{
    int zpageflag; // 0:Bnode,1:Zpage
    struct zpage *zp;
    struct BNode *Bn;
    struct LinearRegression *lr;
    /* data */
} CITLeaf;
map<string, CITLeaf *> str2memleaf;
map<int, long> qid2TrueNumber;
map<int, int> layer1N;
map<string, LinearRegression *> msl;
uniform_real_distribution<float> u(0.0, 1.0);
typedef struct Querys
{
    Query *Qs;
    int queryNumber;
} Querys;
vector<string> leafnames;
map<string, int> leafidx;
int *tolbits = new int;
float midlle[30];
int infL = 30;
int linearkeyLen = 30;
string outputfilepath;
string leafSubname = "";
float pErrorcalculate(int est, int gt)
{
    if (est < gt)
    {
        int t = gt;
        gt = est;
        est = t;
    }
    if (gt == 0)
    {
        gt = 1;
    }
    return (est + 0.0) / (gt + 0.0);
}

void longlong2digVec(long long valx, int *vx, int diglen)
{
    for (int i = 0; i < diglen; i++)
    {
        vx[diglen - i - 1] = (valx % 2);
        valx /= 2;
    }
    if (valx != 0)
    { // overflow
        // cout << "overflow" << endl;
        for (int i = 0; i < diglen; i++)
        {
            vx[diglen - i - 1] = 1;
        }
    }
}

CITLeaf *xreadleafhead(string leafname, int readhead)
{

    string basename = "./data/leaf/";
    // cout << basename + leafname << endl;
    ifstream infile(basename + leafname);
    if (!infile.is_open())
    {

        cout << "Fail to load" << endl;
        // exit(1);
        return NULL;
    }
    int r, state;
    infile >> r >> state;
    // cout<<basename + leafname<<endl;
    CITLeaf *CIL = new CITLeaf;
    if (state == 0) // nonleaf
    {
        if (readhead == 1)
        {
            return CIL;
        }
        // cout<<"READLEAF"<<endl;
        CIL->Bn = NULL;
        CIL->lr = new LinearRegression;
        CIL->zpageflag = 0;
        CIL->zp = NULL;
        CIL->lr->lens = r;
        CIL->lr->kup = new double[r];
        CIL->lr->kdown = new double[r];
        CIL->lr->keys = new long long[r];
        CIL->lr->rows = new long[r];
        for (int i = 0; i < r; i++)
        {
            infile >> CIL->lr->kup[i] >> CIL->lr->kdown[i] >> CIL->lr->keys[i] >> CIL->lr->rows[i];
        }

        // infile >> (CIL->lr->subN);
        // CIL->lr->xlen = r;
        // CIL->lr->coefs = new float[r];
        // for (int i = 0; i < r; i++)
        // {
        //     infile >> (CIL->lr->coefs[i]);
        // }
        // infile >> CIL->lr->intercept;
        // int namel, digl;
        // infile >> namel >> digl;
        // for (int i = 0; i < namel; i++)
        // {
        //     infile >> CIL->Bn->childNumber[i];
        // }
        // for (int i = 0; i < namel; i++)
        // {
        //     for (int j = 0; j < digl; j++)
        //     {
        //         infile >> CIL->Bn->Zvalues[i][j];
        //     }
        // }
    }
    else
    {
        CIL->zpageflag = 1;
        CIL->Bn = NULL;
        CIL->zp = new zpage;
        int digs;
        infile >> digs;
        CIL->zp->digs = digs;
        // CIL->zp->minz = new bool[digs];
        // CIL->zp->maxz = new bool[digs];
        if (readhead == 1)
        {
            for (int i = 0; i < digs; i++)
            {
                infile >> CIL->zp->minz[i];
            }
            for (int i = 0; i < digs; i++)
            {
                infile >> CIL->zp->maxz[i];
            }
            return CIL;
        }
        int r, c;
        infile >> r >> c;
        CIL->zp->r = r;
        CIL->zp->c = c;
        CIL->zp->data = new long long[r * c];
        for (int i = 0; i < r; i++)
        {
            for (int j = 0; j < c; j++)
            {
                long long vx;
                infile >> vx;
                CIL->zp->data[i * c + j] = vx;
            }
        }
    }
    // cout<<CIL->zpageflag<<endl;
    // for(int i=0;i<57;i++){
    //     cout<<CIL->Bn->Zvalues[0][i]<<" ";
    // }cout<<endl;
    // cout<<CIL->zp->r<<" "<<CIL->zp->c<<endl;
    // cout<<CIL->zp->data[((CIL->zp->r)-1)*CIL->zp->c+1]<<" "<< CIL->zp->digs<<endl;
    // cout<<"D"<<endl;
    return CIL;
}

CITLeaf *readleaf(string leafname)
{
    if (str2memleaf[leafname] != 0)
    {
        return str2memleaf[leafname];
    }

    string basename = "./data/leaf" + leafSubname + "/";
    // cout << basename + leafname << endl;
    ifstream infile(basename + leafname);
    if (!infile.is_open())
    {

        // cout << "Fail to load" << endl;
        // exit(1);
        return NULL;
    }
    int r, state;
    infile >> r >> state;
    // cout<<basename + leafname<<endl;
    CITLeaf *CIL = new CITLeaf;
    if (state == 0) // nonleaf
    {
        CIL->Bn = NULL;
        CIL->lr = new LinearRegression;
        CIL->zpageflag = 0;
        CIL->zp = NULL;
        CIL->lr->lens = r;
        CIL->lr->kup = new double[r];
        CIL->lr->kdown = new double[r];
        CIL->lr->keys = new long long[r];
        CIL->lr->rows = new long[r];
        for (int i = 0; i < r; i++)
        {
            // cout<<"Read"<<" ";
            infile >> CIL->lr->kup[i] >> CIL->lr->kdown[i] >> CIL->lr->keys[i] >> CIL->lr->rows[i];
            // cout<<CIL->lr->kup[i]<< " "<<CIL->lr->keys[i]<<endl;
        }

        // infile >> (CIL->lr->subN);
        // CIL->lr->xlen = r;
        // CIL->lr->coefs = new float[r];
        // for (int i = 0; i < r; i++)
        // {
        //     infile >> (CIL->lr->coefs[i]);
        // }
        // infile >> CIL->lr->intercept;
        // int namel, digl;
        // infile >> namel >> digl;
        // for (int i = 0; i < namel; i++)
        // {
        //     infile >> CIL->Bn->childNumber[i];
        // }
        // for (int i = 0; i < namel; i++)
        // {
        //     for (int j = 0; j < digl; j++)
        //     {
        //         infile >> CIL->Bn->Zvalues[i][j];
        //     }
        // }
    }
    else
    {
        CIL->zpageflag = 1;
        CIL->Bn = NULL;
        CIL->zp = new zpage;
        int digs;
        infile >> digs;
        CIL->zp->digs = digs;
        // CIL->zp->minz = new bool[digs];
        // CIL->zp->maxz = new bool[digs];
        for (int i = 0; i < digs; i++)
        {
            infile >> CIL->zp->minz[i];
        }
        for (int i = 0; i < digs; i++)
        {
            infile >> CIL->zp->maxz[i];
        }
        int r, c;
        infile >> r >> c;
        CIL->zp->r = r;
        CIL->zp->c = c;
        CIL->zp->data = new long long[r * c];
        for (int i = 0; i < r; i++)
        {
            for (int j = 0; j < c; j++)
            {
                long long vx;
                infile >> vx;
                CIL->zp->data[i * c + j] = vx;
            }
        }
    }
    // cout<<CIL->zpageflag<<endl;
    // for(int i=0;i<57;i++){
    //     cout<<CIL->Bn->Zvalues[0][i]<<" ";
    // }cout<<endl;
    // cout<<CIL->zp->r<<" "<<CIL->zp->c<<endl;
    // cout<<CIL->zp->data[((CIL->zp->r)-1)*CIL->zp->c+1]<<" "<< CIL->zp->digs<<endl;
    // cout<<"D"<<endl;
    return CIL;
}
long long nonleafSize = 0;
long long leafSize = 0;
long long rootCoreSize = 0;

MADENet *loadMade(string filePath)
{
    ifstream infile(filePath);
    if (!infile.is_open())
    {
        cout << "Fail to Net load Tree" << endl;
        return NULL;
    }
    int bittol;
    // infile >> bittol;
    // cout<<"bt:"<<bittol<<endl;
    MADENet *ret = new MADENet;
    nonleafSize += (sizeof(MADENet));
    rootCoreSize += (sizeof(MADENet));
    infile >> ret->zdr >> ret->zdc >> ret->connectlen >> ret->leafnums;
    bittol = ret->zdc;
    ret->diglen = bittol;
    ret->fc1w = new float[bittol * bittol];
    ret->fc2w = new float[bittol * bittol];
    ret->fc1b = new float[bittol];
    ret->fc2b = new float[bittol];
    nonleafSize += (2 * (bittol * bittol * 4) + 2 * bittol * 4);
    rootCoreSize += (2 * (bittol * bittol * 4) + 2 * bittol * 4);

    for (int i = 0; i < bittol; i++)
    {
        for (int j = 0; j < bittol; j++)
        {
            infile >> ret->fc1w[i * bittol + j];
        }
    }
    for (int i = 0; i < bittol; i++)
    {
        infile >> ret->fc1b[i];
    }
    for (int i = 0; i < bittol; i++)
    {
        for (int j = 0; j < bittol; j++)
        {
            infile >> ret->fc2w[i * bittol + j];
        }
    }
    for (int i = 0; i < bittol; i++)
    {
        infile >> ret->fc2b[i];
    }
    int strcord = 0;
    // return ret;
    cout << "Loading Nodes" << endl;

    for (int i = 0; i < ret->leafnums; i++)
    {
        cout << "\rLoading pct:" << i << ' ';
        layer1N[i] = 0;
        CITLeaf *ldn = readleaf("N-" + to_string(i));
        if (ldn != NULL)
        {
            nonleafSize += sizeof(CITLeaf);
            if (ldn->zpageflag == 0)
            {
                nonleafSize += sizeof(LinearRegression);
                nonleafSize += (2 * sizeof(double) * (ldn->lr->lens));
                nonleafSize += (sizeof(long long) * (ldn->lr->lens) + sizeof(long) * (ldn->lr->lens));
                msl["N-" + to_string(i)] = ldn->lr;
            }
            else
            {
                msl["N-" + to_string(i)] = NULL;
            }
            leafnames.push_back("N-" + to_string(i));
            leafidx["N-" + to_string(i)] = strcord;
            strcord += 1;
            layer1N[i] = 1;
        }
        else
        {
            leafidx["N-" + to_string(i)] = -1;
        }
    }
    // return ret;
    cout << "L2" << endl;
    for (int i = 0; i < ret->leafnums; i++)
    {
        cout << "\rLoading pct:" << i << ' ';
        string namex = "N-" + to_string(i);
        CITLeaf *ldn = readleaf(namex);
        str2memleaf[namex] = ldn;
        for (int j = 0; j < 4000; j++)
        {
            string name = "N-" + to_string(i) + "-" + to_string(j);
            CITLeaf *ldn = readleaf(name);
            str2memleaf[name] = ldn;
            if (ldn == NULL)
            {
                break;
            }
        }
    }
    cout << leafnames.size() << endl;
    cout << "Load Done" << strcord << endl;

    return ret;
}
void MadeIndexInferDig(int *xinput, float *out, int startidx, int endidx, MADENet *net, float *middle)
{
    int winlen = net->connectlen;
    for (int i = startidx; i <= endidx; i++)
    {
        middle[i] = net->fc1b[i];
        for (int j = max(i - winlen, 0); j < i; j++)
        {
            if (j >= i)
            {
                break;
            }
            middle[i] += (xinput[j] * net->fc1w[i * net->diglen + j]);
        }
        if (middle[i] < 0)
        {
            middle[i] = 0;
        }
    }
    // for (int i=0;i<5;i++){
    //     cout<<middle[i]<<" ";
    // }cout<<endl;
    for (int i = startidx; i <= endidx; i++)
    {
        out[i] = net->fc2b[i];
        // cout<<"oi"<<out[i]<<endl;
        for (int j = max(i - winlen, 0); j < i; j++)
        {
            if (j >= i)
            {
                break;
            }
            out[i] += (middle[j] * net->fc2w[i * net->diglen + j]);
        }
        // cout<<out[i]<<endl;
        out[i] = (1.0) / (1.0 + exp(-out[i]));
        // cout<<out[i]<<endl;
    }
    // for(int i=0;i<5;i++){
    //     cout<<out[i]<<endl;
    // }
    // exit(1);
}
void MadeIndexInfer(int *xinput, float *out, int preLen, MADENet *net, float *middle)
{
    int winlen = net->connectlen;
    for (int i = 0; i < preLen; i++)
    {
        middle[i] = net->fc1b[i];
        // cout<<middle[i]<<endl;
        for (int j = max(i - winlen, 0); j < i; j++)
        {
            // cout<<"j"<<j<<endl;
            if (j >= i)
            {
                break;
            }
            // cout<<"xij: "<<xinput[j]<<endl;
            middle[i] += (xinput[j] * net->fc1w[i * net->diglen + j]);
        }
        if (middle[i] < 0)
        {
            middle[i] = 0;
        }
    }
    // for(int i=0;i<preLen;i++){
    //     cout<<middle[i]<<" ";
    // }cout<<endl;
    for (int i = 0; i < preLen; i++)
    {
        out[i] = net->fc2b[i];
        for (int j = max(i - winlen, 0); j < i; j++)
        {
            if (j >= i)
            {
                break;
            }
            out[i] += (middle[j] * net->fc2w[i * net->diglen + j]);
        }
        out[i] = (1.0) / (1.0 + exp(-out[i]));
    }
}
void testInfer()
{
    typedef std::chrono::high_resolution_clock Clock;
    MADENet *M = loadMade("./Model/MadeRoot.txt");
    int *xinput = new int[300];
    float *out = new float[300];
    long long cnt = 0;
    int ucnt = 2500000;
    float middle[200];
    for (int i = 0; i < ucnt; i++)
    {

        auto t1 = Clock::now(); // 计时开始

        for (int ix = 0; ix < 30; ix++)
        {
            xinput[ix] = 0;
            middle[ix] = 0;
        }
        MadeIndexInfer(xinput, out, 20, M, middle);
        auto t2 = Clock::now(); // 计时开始
        cnt += (std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count());
    }
    cout << cnt / ucnt << endl;
}

Querys *readQueryFile(string queryfilename)
{
    ifstream infile(queryfilename);
    if (!infile.is_open())
    {
        cout << queryfilename << endl;
        cout << "Fail to load" << endl;
        return NULL;
    }
    int colNumber, queryNumber;
    infile >> colNumber >> queryNumber;
    // cout<<colNumber<< "  "<<queryNumber<<endl;
    Querys *A = new Querys;
    A->queryNumber = queryNumber;
    A->Qs = new Query[queryNumber];
    int *binaryLength = new int[queryNumber];
    for (int i = 0; i < colNumber; i++)
    {
        infile >> binaryLength[i];
    }
    for (int i = 0; i < queryNumber; i++)
    {
        A->Qs[i].binaryLength = binaryLength;
        A->Qs[i].columnNumber = colNumber;
        A->Qs[i].queryid = i;
        A->Qs[i].leftupBound = new long long[colNumber];
        A->Qs[i].rightdownBound = new long long[colNumber];
        for (int j = 0; j < colNumber; j++)
        {
            infile >> A->Qs[i].leftupBound[j];
        }
        for (int j = 0; j < colNumber; j++)
        {
            infile >> A->Qs[i].rightdownBound[j];
        }
        long Tnumber;
        infile >> Tnumber;
        qid2TrueNumber[i] = Tnumber;
        // cout<<Tnumber<<endl;
    }
    // cout << "end" << endl;
    return A;
}

int *QueryUp2Zvalue(Query Qi, int *tolbitsx, int rightupflag)
{
    // 将Qi的端点转Z值。同时首位用0填充,rightupflag=1：左上，=0,右下
    vector<vector<int>> bCs;
    // for(int i=0;i<45;i++){
    //     cout<<1;
    // }exit(1);
    for (int i = 0; i < Qi.columnNumber; i++)
    {
        vector<int> bvi;
        int digitTollen = Qi.binaryLength[i];
        int *vi = new int[digitTollen + 1];

        long long v;
        if (rightupflag == 0)
        {
            v = Qi.leftupBound[i];
        }
        else
        {
            v = Qi.rightdownBound[i];
        }
        longlong2digVec(v, vi, digitTollen);

        for (int ix = 0; ix < digitTollen; ix += 1)
        {
            bvi.push_back(vi[ix]);
        }
        // cout << endl
        //      << v << " " << digitTollen << endl;
        bCs.push_back(bvi);
    }
    int tolbits = 0;
    int maxbit = -1;
    for (int i = 0; i < Qi.columnNumber; i++)
    {
        if (Qi.binaryLength[i] > maxbit)
        {
            maxbit = Qi.binaryLength[i];
        }
        tolbits += Qi.binaryLength[i];
    }
    *tolbitsx = tolbits;
    int *zencode = new int[tolbits + 1];
    zencode[0] = 0;
    int cnt = 1;
    for (int i = 0; i < maxbit; i++)
    {
        for (int j = 0; j < Qi.columnNumber; j++)
        {
            if (i >= Qi.binaryLength[j])
            {
                continue;
            }
            else
            {
                zencode[cnt] = bCs[j][i];
                cnt += 1;
            }
        }
    }
    return zencode;
}
float point2cdfest(MADENet *root, Query Qi, int *zencode)
{
    float cdf = 0;
    float acc = 1.0;
    float OneProb;
    int belong;
    float minnimumsep;
    float out[30];
    MadeIndexInfer(&zencode[1], out, infL, root, midlle);
    for (int i = 0; i < infL; i++)
    {

        OneProb = out[i];
        cdf += (acc * (1 - OneProb) * zencode[i + 1]);
        acc *= (OneProb * zencode[i + 1] + (1 - OneProb) * (1 - zencode[i + 1]));
        // cout<<i<<OneProb<<" "<<cdf<<" "<<acc<<endl;
    }
    return cdf;
}
int point2blockName(MADENet *root, Query Qi, int *zencode)
{
    float cdf = 0;
    float acc = 1.0;
    float OneProb;
    int belong;
    float minnimumsep;
    float out[30];
    MadeIndexInfer(&zencode[1], out, infL, root, midlle);
    for (int i = 0; i < infL; i++)
    {

        OneProb = out[i];
        cdf += (acc * (1 - OneProb) * zencode[i + 1]);
        acc *= (OneProb * zencode[i + 1] + (1 - OneProb) * (1 - zencode[i + 1]));
        // cout<<i<<OneProb<<" "<<cdf<<" "<<acc<<endl;
    }
    minnimumsep = 1 / (0.0 + root->leafnums);
    belong = cdf / minnimumsep;
    // cout<<cdf<<endl;
    // cout <<belong<<" "<< minnimumsep << endl;
    return belong;
}

void testTimePointQ(MADENet *M, string qfs)
{
    Querys *qs = readQueryFile(qfs);
    // MADENet *M = loadMade("./Model/MadeRoot.txt");
    ofstream ofs(outputfilepath + "PointQ.txt");
    typedef std::chrono::high_resolution_clock Clock;
    int loadCnt = 0;
    int nonleafCnt = 0;
    int scanCnt = 0;
    int o = 0;
    int loopC = 1;
    for (int k = 0; k < loopC; k++)
    {
        // cout << k << endl;
        for (int qix = 0; qix < qs->queryNumber; qix++)
        {
            // qix=1;
            // cout<<"qid:"<<qix<<endl;
            Query qi = qs->Qs[qix];
            // for (int ix = 0; ix < 6; ix++)
            // {
            //     cout << qi.leftupBound[ix] << " ";
            // }
            // cout << endl;
            int *zencode = QueryUp2Zvalue(qi, tolbits, 0);
            auto t1 = Clock::now(); // 计时开始
            int blkidxp = point2blockName(M, qi, zencode);
            auto t2 = Clock::now(); // 计时结束
            string blkfirstname = "N-" + to_string(blkidxp);
            // cout<<blkfirstname<<endl;
            // exit(1);
            nonleafCnt += (std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count());
            // cout << blkfirstname << endl;
            // CITLeaf *CITL = readleaf(blkfirstname);
            CITLeaf *CITL = new CITLeaf;
            // cout<<CITL->zpageflag<<endl;
            CITL->zpageflag = 0;
            CITL->lr = msl[blkfirstname];
            if (CITL->lr == NULL)
            {
                continue;
            }
            // cout<<blkfirstname<<endl;
            // leafNodeData *lnd = readBinary(to_string(blknum0) + "b", 1);
            LinearRegression *lrg = CITL->lr;
            long long keyinput = 0;
            for (int it = 0; it < 64; it++)
            {
                keyinput *= 2;
                keyinput += zencode[it + 1];
            }

            int actr = 0;
            // cout<<"S"<<endl;
            // cout<<CITL->zpageflag<<endl;
            auto t2d = Clock::now(); // 计时结束
            if (CITL->zpageflag == 0)
            {

                // for (int it = 0; it < 93; it++)
                // {
                //     cout << zencode[it];
                // }
                // cout << endl;
                float cdfestl = 0, cdfestr = 0;
                long CDFTol = 0;
                // cout << "Kipt:" << keyinput << endl;
                // cout << "Key1:" << lrg->keys[0] << endl;
                // cout << "Key2:" << lrg->keys[1] << endl;
                for (int it = 0; it < lrg->lens; it++)
                {
                    CDFTol += lrg->rows[it];
                }
                auto t3 = Clock::now(); // 计时结束
                // cout<<CDFTol<<endl;
                long basecdf = 0;
                // cout << lrg->lens << endl;
                float bincdf = 100.0 / CDFTol;
                for (int it = 0; it < lrg->lens; it++)
                {
                    // cout<<lrg->keys[it ]<<endl;
                    if (lrg->keys[it] > keyinput)
                    {
                        basecdf -= lrg->rows[it - 1];

                        cdfestl = -bincdf * 0.1 + (basecdf + 0.0) / CDFTol + (keyinput - (lrg->keys[it - 1])) * (lrg->kdown[it - 1]);
                        cdfestr = bincdf * 0.1 + (basecdf + 0.0) / CDFTol + (keyinput - (lrg->keys[it - 1])) * (lrg->kup[it - 1]);
                        break;
                    }
                    basecdf += lrg->rows[it];
                }
                int lastx = (lrg->lens) - 1;
                // cout<<CDFTol<<endl;
                // cout<<keyinput<<endl;
                if (keyinput > lrg->keys[lastx])
                {
                    // cout << "last round" << endl;basecdf -= lrg->rows[lastx];
                    // cout << (basecdf + 0.0) / CDFTol << endl;
                    // cout<<bincdf*0.2<<endl;
                    // cout << basecdf << endl;
                    cdfestl = -bincdf * 0.1 + (basecdf + 0.0) / CDFTol + (keyinput - (lrg->keys[lastx])) * (lrg->kdown[lastx]);
                    cdfestr = bincdf * (0.1) + (basecdf + 0.0) / CDFTol + (keyinput - (lrg->keys[lastx])) * (lrg->kup[lastx]);
                }
                auto t4 = Clock::now(); // 计时结束

                scanCnt += (std::chrono::duration_cast<std::chrono::nanoseconds>(t4 - t3).count());
                // cdfestl = ( 0.0) / CDFTol + (keyinput - (lrg->keys[0 ])) * (lrg->kdown[0 ]) - bincdf*0.2;
                // cdfestr = (  0.0) / CDFTol + (keyinput - (lrg->keys[0 ])) * (lrg->kup[0 ])+ bincdf *0.2;
                // exit(1);
                // cout << cdfestl << " " << cdfestr << endl;
                // cout<<cdfestl<<" "<<(cdfestl/bincdf)<<endl;
                if (cdfestl >= 1)
                {
                    cdfestl = 1.0;
                }
                if (cdfestr >= 1)
                {
                    cdfestr = 1.0;
                }
                if (cdfestl <= 0)
                {
                    cdfestl = 0;
                }
                if (cdfestr <= 0)
                {
                    cdfestr = 0;
                }

                // cout << bincdf << endl;
                int belong1 = cdfestl / bincdf, belong2 = cdfestr / bincdf;
                // blkfirstname += ("-" + to_string(belong1));
                int findflag = 0;
                // cout<<belong1<<" "<<belong2<<endl;
                for (int jx = belong1; jx <= belong2; jx++)
                {
                    // cout<<"Read:"<<blkfirstname + ("-" + to_string(jx))<<end;
                    CITL = readleaf(blkfirstname + ("-" + to_string(jx)));
                    if (CITL == NULL)
                    {
                        continue;
                    }
                    int rx, cx;
                    bool *maxptr;
                    bool *minptr;
                    maxptr = CITL->zp->maxz;
                    minptr = CITL->zp->minz;
                    int bigthanL = 0, smalthanR = 0;
                    int alwayseq0 = 0, alwayseq1 = 0;
                    for (int i = 0; i < 40; i++)
                    {
                        // cout << "I:" << i << " " << minptr[i] << " " << zencode[i + 1] << endl;
                        if (minptr[i] == zencode[i + 1])
                        {
                            alwayseq0 = 1;
                            continue;
                        }
                        else if (minptr[i] < zencode[i + 1])
                        {
                            alwayseq0 = 0;
                            bigthanL = 1;
                            break;
                        }
                        else if (minptr[i] > zencode[i + 1])
                        {
                            alwayseq0 = 0;
                            bigthanL = -1;
                            break;
                        }
                    }
                    for (int i = 0; i < 40; i++)
                    {
                        if (maxptr[i] == zencode[i + 1])
                        {
                            alwayseq1 = 1;
                            continue;
                        }
                        else if (maxptr[i] < zencode[i + 1])
                        {
                            alwayseq1 = 0;
                            smalthanR = -1;
                            break;
                        }
                        else if (maxptr[i] > zencode[i + 1])
                        {
                            alwayseq1 = 0;
                            smalthanR = 1;
                            break;
                        }
                    }
                    if (alwayseq1 == 1)
                    {
                        smalthanR = 1;
                    }
                    if (alwayseq0 = 1)
                    {
                        bigthanL = 1;
                    }
                    // cout << bigthanL << " " << smalthanR << endl;
                    if (smalthanR == 1 && bigthanL == 1)
                    {
                        findflag = 1;
                        // cout<<"find"<<blkfirstname + ("-" + to_string(jx))<<endl;
                        break;
                    }
                }

                // if (findflag == 0)
                // {
                //     // cout << " Cannot find!" << endl;
                //     // exit(1);
                // }
                // CITL = readleaf(blkfirstname + ("-" + to_string(belong1)));

                // cout<<"Bel"<<belong<<endl;
                // t3=t2;
                // BNode *bn = CITL->Bn;
                // for (int it = 0; it < bn->itemN; it++)
                // {
                //     int findFlag = 0;
                //     for (int j = 0; j < 40; j++)
                //     {
                //         if (zencode[j + 1] > bn->Zvalues[it][j])
                //         {
                //             findFlag = 1;
                //             break;
                //         }
                //     }
                //     if (findFlag == 1)
                //     {

                //         actr = bn->childNumber[i];
                //         break;
                //     }
                // }

                // if( CITL ==NULL)
            }

            o += actr;
            auto t3 = Clock::now(); // 计时结束
            loadCnt += (std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2d).count());

            // loadCnt += (std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count());
            // exit(1);
        }

        // exit(1);
    }
    // cout<<"ff"<<o<<endl;
    ofs << "In average:\n"
        << "Noneleaf:" << nonleafCnt / (qs->queryNumber * loopC) << " Load:" << loadCnt / (qs->queryNumber * loopC) << " Scan: " << scanCnt / (qs->queryNumber * loopC) << endl;

    cout << "In average:\n"
         << "Noneleaf:" << nonleafCnt / (qs->queryNumber * loopC) << " Load:" << loadCnt / (qs->queryNumber * loopC) << " Scan: " << scanCnt / (qs->queryNumber * loopC) << endl;
}

int *pointQueryTriple(MADENet *root, Query Qi, int *zencode)
{
    // 利用Fiting线性，走完最后一程

    int first2 = point2blockName(root, Qi, zencode);
    int *ret = new int[3];
    ret[0] = first2;
    string lrkey = "N-" + to_string(ret[0]);
    if (msl.find(lrkey) == msl.end())
    {
        ret[1] = -1;
        return ret;
    }
    if (msl[lrkey] == NULL)
    {
        ret[1] = -1;
        return ret;
    }
    CITLeaf *cil = readleaf(lrkey);
    // cil =
    // cout<<lrkey<<linearkeyLen<<endl;
    LinearRegression *lrg = cil->lr;
    long long keyinput = 0;
    for (int it = 0; it < linearkeyLen; it++)
    {
        keyinput *= 2;
        keyinput += zencode[it + 1];
    }
    double cdfestl = 0, cdfestr = 0;
    long CDFTol = 0;
    for (int it = 0; it < lrg->lens; it++)
    {
        CDFTol += lrg->rows[it];
    }

    long basecdf = 0;
    // cout<<"KI "<<keyinput<<endl;
    // cout << lrg->lens << endl;
    double bincdf = 100.0;
    double pagecdf = bincdf;
    // cout<<bincdf<<endl;
    // cout<<lrkey<<endl;
    // cout<<lrg->lens<<" "<<lrg->rows<<endl;
    // for(int i=0;i<3;i++){
    //     cout<<lrg->keys[i]<<" ";
    // }cout<<endl;
    // for(int i=0;i<3;i++){
    //     cout<<lrg->kdown[i]<<" ";
    // }cout<<endl;

    double errorb = bincdf * 0.1;
    // cout<<errorb<<" "<<bincdf<<endl;
    for (int it = 0; it < lrg->lens; it++)
    {
        // cout<<"LRK: "<<lrg->keys[it ]<<endl;
        if (lrg->keys[it] > keyinput)
        {
            // cout<<"big "<<endl;

            basecdf -= lrg->rows[it - 1];
            // cout<<lrg->keys[it]<<' '<<keyinput<<endl;
            // cout<<"bc"<<basecdf<<endl;
            // cout<<(basecdf + 0.0) / CDFTol<<" "<<(keyinput - (lrg->keys[it - 1])) * (lrg->kdown[it - 1])<<endl;
            cdfestl = -errorb + (basecdf + 0.0) + (keyinput - (lrg->keys[it - 1])) * (lrg->kdown[it - 1]);
            cdfestr = errorb + (basecdf + 0.0) + (keyinput - (lrg->keys[it - 1])) * (lrg->kup[it - 1]);
            break;
        }
        basecdf += lrg->rows[it];
    }
    int lastx = (lrg->lens) - 1;
    // cout<<CDFTol<<" "<<basecdf<<endl;
    // cout<<keyinput<<endl;
    // cout<<cdfestl<<" "<<cdfestr<<endl;
    if (keyinput > lrg->keys[lastx])
    {
        // cout << "last round" << endl;
        basecdf -= lrg->rows[lastx];
        // cout << (basecdf + 0.0) / CDFTol << endl;
        // cout<<bincdf*0.2<<endl;
        // cout << basecdf << endl;
        cdfestl = -errorb + (basecdf + 0.0) + (keyinput - (lrg->keys[lastx])) * (lrg->kdown[lastx]);
        cdfestr = errorb + (basecdf + 0.0) + (keyinput - (lrg->keys[lastx])) * (lrg->kup[lastx]);
    }
    // cdfestl = ( 0.0) / CDFTol + (keyinput - (lrg->keys[0 ])) * (lrg->kdown[0 ]) - bincdf*0.2;
    // cdfestr = (  0.0) / CDFTol + (keyinput - (lrg->keys[0 ])) * (lrg->kup[0 ])+ bincdf *0.2;
    // exit(1);
    // cout << cdfestl << " " << cdfestr << endl;
    // cout<<(cdfestl/bisncdf)<<endl;
    // if (cdfestl >= 1)
    // {
    //     cdfestl = 1.0;
    // }
    // if (cdfestr >= 1)
    // {
    //     cdfestr = 1.0;
    // }
    // if (cdfestl <= 0)
    // {
    //     cdfestl = 0;
    // }
    // if (cdfestr <= 0)
    // {
    //     cdfestr = 0;
    // }

    // cout << bincdf << endl;
    // cout<<cdfestl<<" "<<cdfestr<<endl;
    int belong1 = cdfestl / pagecdf, belong2 = cdfestr / pagecdf;
    // blkfirstname += ("-" + to_string(belong1));
    int findflag = 0;
    // cout<<"bl: "<<belong1<<" "<<belong2<<endl;
    ret[1] = (belong1);
    ret[2] = belong2;
    return ret;
    // exit(1);
    for (int jx = belong1; jx <= belong2; jx++)
    {
        CITLeaf *CITL = readleaf(lrkey + ("-" + to_string(jx)));
        if (CITL == NULL)
        {
            continue;
        }
        int rx, cx;
        bool *maxptr;
        bool *minptr;
        maxptr = CITL->zp->maxz;
        minptr = CITL->zp->minz;
        int bigthanL = 0, smalthanR = 0;
        int alwayseq0 = 0, alwayseq1 = 0;
        for (int i = 0; i < 40; i++)
        {
            // cout << "I:" << i << " " << minptr[i] << " " << zencode[i + 1] << endl;
            if (minptr[i] == zencode[i + 1])
            {
                alwayseq0 = 1;
                continue;
            }
            else if (minptr[i] < zencode[i + 1])
            {
                alwayseq0 = 0;
                bigthanL = 1;
                break;
            }
            else if (minptr[i] > zencode[i + 1])
            {
                alwayseq0 = 0;
                bigthanL = -1;
                break;
            }
        }
        for (int i = 0; i < 40; i++)
        {
            if (maxptr[i] == zencode[i + 1])
            {
                alwayseq1 = 1;
                continue;
            }
            else if (maxptr[i] < zencode[i + 1])
            {
                alwayseq1 = 0;
                smalthanR = -1;
                break;
            }
            else if (maxptr[i] > zencode[i + 1])
            {
                alwayseq1 = 0;
                smalthanR = 1;
                break;
            }
        }
        if (alwayseq1 == 1)
        {
            smalthanR = 1;
        }
        if (alwayseq0 = 1)
        {
            bigthanL = 1;
        }
        // cout << bigthanL << " " << smalthanR << endl;
        if (smalthanR == 1 && bigthanL == 1)
        {
            // cout<<"FD"<<endl;
            findflag = 1;
            ret[1] = jx;
            return ret;
            // break;
        }
    }
    // cout<<"over"<<endl;
    return ret;
}
bool *getLITMAX(bool *minZ, bool *maxZ, bool *zvalue, int bitlength, int *colbitlength, int colNum)
{
    bool *Litmax = new bool[bitlength];
    bool *tmpmaxZ = new bool[bitlength];
    bool *tmpminZ = new bool[bitlength];
    for (int i = 0; i < bitlength; i += 1)
    {
        tmpminZ[i] = minZ[i];
        tmpmaxZ[i] = maxZ[i];
        Litmax[i] = 0;
    }
    int maxlen = 0;
    for (int i = 0; i < colNum; i += 1)
    {
        if (colbitlength[i] > maxlen)
        {
            maxlen = colbitlength[i];
        }
    }
    int idx = 0;
    for (int i = 0; i < maxlen; i += 1)
    {
        for (int j = 0; j < colNum; j++)
        {
            if (i >= colbitlength[j])
            {
                continue;
            }
            else
            {
                // cout<<i<<"th dig of col"<<j<<endl;
                int divnum = zvalue[idx];
                int minnum = tmpminZ[idx];
                int maxnum = tmpmaxZ[idx];
                // cout << "Idx" << idx << " div:" << divnum << " minnum:" << minnum << "  maxN:" << maxnum << endl;
                // cout << "Tminz:";
                // for (int i = 0; i < bitlength; i++)
                // {
                //     cout << tmpminZ[i] << " ";
                // }
                // cout << endl;
                // cout << "Tmaxz:";
                // for (int i = 0; i < bitlength; i++)
                // {
                //     cout << tmpmaxZ[i] << " ";
                // }
                // cout << endl;
                // cout << "Bigmi:";
                // for (int i = 0; i < bitlength; i++)
                // {
                //     cout << Bigmin[i] << " ";
                // }
                // cout << endl;

                if (divnum == 0 && minnum == 0 && maxnum == 0)
                {
                    idx += 1;
                    continue;
                }
                if (divnum == 1 && minnum == 1 && maxnum == 1)
                {
                    idx += 1;
                    continue;
                }
                if (divnum == 0 && minnum == 1 && maxnum == 1)
                {
                    // cout<<"code:011"<<endl;
                    return Litmax;
                }
                if (divnum == 1 && minnum == 0 && maxnum == 0)
                {
                    // cout<<"code100"<<endl;
                    return tmpmaxZ;
                }
                if (divnum == 0 && minnum == 1 && maxnum == 0)
                {
                    return zvalue;
                    cout << "LITMAXWRONG!" << endl;
                    exit(1);
                    // return Bigmin;
                }
                if (divnum == 1 && minnum == 1 && maxnum == 0)
                {
                    return zvalue;
                    cout << "LITMAXWRONG!" << endl;
                    exit(1);
                    // return Bigmin;
                }

                if (divnum == 0 && minnum == 0 && maxnum == 1)
                {
                    // cout << "CODE: 101" << endl;
                    // max = 1000000
                    int innercnt = 0;
                    for (int x0 = 0; x0 < maxlen; x0++)
                    {
                        for (int x1 = 0; x1 < colNum; x1 += 1)
                        {
                            if (x0 >= colbitlength[x1])
                            {
                                continue;
                            }
                            else
                            {
                                if (x0 < i)
                                {
                                    innercnt++;
                                    continue;
                                }
                                else if (x0 == i)
                                {
                                    if (x1 == j)
                                    {
                                        tmpmaxZ[innercnt] = 0;
                                    }
                                    innercnt += 1;
                                }
                                else
                                {
                                    if (x1 == j)
                                    {
                                        tmpminZ[innercnt] = 1;
                                    }
                                    innercnt += 1;
                                }
                            }
                        }
                    }
                    // Bigmin = tmpminZ;
                    idx += 1;
                    continue;
                }
                if (divnum == 1 && minnum == 0 && maxnum == 1)
                {
                    // cout << "CODE:001" << endl;
                    for (int x00 = 0; x00 < bitlength; x00++)
                    {
                        Litmax[x00] = tmpmaxZ[x00];
                    }
                    int innercnt = 0;
                    for (int x0 = 0; x0 < maxlen; x0++)
                    {
                        for (int x1 = 0; x1 < colNum; x1 += 1)
                        {
                            if (x0 >= colbitlength[x1])
                            {
                                continue;
                            }
                            else
                            {
                                if (x0 < i)
                                {
                                    innercnt++;
                                    continue;
                                }
                                else if (x0 == i)
                                {
                                    if (x1 == j)
                                    {
                                        Litmax[innercnt] = 0;
                                    }
                                    innercnt += 1;
                                }
                                else
                                {
                                    if (x1 == j)
                                    {
                                        Litmax[innercnt] = 1;
                                    }
                                    innercnt += 1;
                                }
                            }
                        }
                    }
                    innercnt = 0;
                    for (int x0 = 0; x0 < maxlen; x0++)
                    {
                        for (int x1 = 0; x1 < colNum; x1 += 1)
                        {
                            if (x0 >= colbitlength[x1])
                            {
                                continue;
                            }
                            else
                            {
                                if (x0 < i)
                                {
                                    innercnt++;
                                    continue;
                                }
                                else if (x0 == i)
                                {
                                    if (x1 == j)
                                    {
                                        tmpminZ[innercnt] = 1;
                                    }
                                    innercnt += 1;
                                }
                                else
                                {
                                    if (x1 == j)
                                    {
                                        tmpminZ[innercnt] = 0;
                                    }
                                    innercnt += 1;
                                }
                            }
                        }
                    }
                    // cout << "Tminz:";
                    // for (int i = 0; i < bitlength; i++)
                    // {
                    //     cout << tmpminZ[i];
                    // }
                    // cout << endl;
                    // cout << "Tmaxz:";
                    // for (int i = 0; i < bitlength; i++)
                    // {
                    //     cout << tmpmaxZ[i];
                    // }
                    // cout << endl;
                    // cout << "Bigmi:";
                    // for (int i = 0; i < bitlength; i++)
                    // {
                    //     cout << Bigmin[i];
                    // }
                    // cout << endl;
                    idx += 1;
                    continue;
                }
                idx += 1;
            }
        }
    }
    // cout<<"Normal Ret"<<endl;
    return Litmax;
}

bool *getBIGMIN(bool *minZ, bool *maxZ, bool *zvalue, int bitlength, int *colbitlength, int colNum)
{
    // 输入：查询框的左上minz，右下maxz，范围外的zvalue，返回zvalue的bigmin

    bool *Bigmin = new bool[bitlength];
    bool *tmpmaxZ = new bool[bitlength];
    bool *tmpminZ = new bool[bitlength];
    for (int i = 0; i < bitlength; i += 1)
    {
        tmpminZ[i] = minZ[i];
        tmpmaxZ[i] = maxZ[i];
        Bigmin[i] = 0;
    }
    // cout << "iptzv:";
    // for (int i = 0; i < bitlength - 1; i++)
    // {
    //     cout << zvalue[i];
    // }
    // cout << endl;
    // cout << "Tminz:";
    // for (int i = 0; i < bitlength - 1; i++)
    // {
    //     cout << tmpminZ[i];
    // }
    // cout << endl;
    // cout << "Tmaxz:";
    // for (int i = 0; i < bitlength - 1; i++)
    // {
    //     cout << tmpmaxZ[i];
    // }
    // cout << endl;

    int maxlen = 0;
    for (int i = 0; i < colNum; i += 1)
    {
        if (colbitlength[i] > maxlen)
        {
            maxlen = colbitlength[i];
        }
    }
    int idx = 0;
    for (int i = 0; i < maxlen; i += 1)
    {
        for (int j = 0; j < colNum; j++)
        {
            if (i >= colbitlength[j])
            {
                continue;
            }
            else
            {
                // cout<<i<<"th dig of col"<<j<<endl;
                int divnum = zvalue[idx];
                int minnum = tmpminZ[idx];
                int maxnum = tmpmaxZ[idx];
                // cout << "Idx" << idx << " div:" << divnum << " minnum:" << minnum << "  maxN:" << maxnum << endl;
                // cout << "Tminz:";
                // for (int i = 0; i < bitlength; i++)
                // {
                //     cout << tmpminZ[i] << " ";
                // }
                // cout << endl;
                // cout << "Tmaxz:";
                // for (int i = 0; i < bitlength; i++)
                // {
                //     cout << tmpmaxZ[i] << " ";
                // }
                // cout << endl;
                // cout << "Bigmi:";
                // for (int i = 0; i < bitlength; i++)
                // {
                //     cout << Bigmin[i] << " ";
                // }
                // cout << endl;

                if (divnum == 0 && minnum == 0 && maxnum == 0)
                {
                    idx += 1;
                    continue;
                }
                if (divnum == 1 && minnum == 1 && maxnum == 1)
                {
                    idx += 1;
                    continue;
                }
                if (divnum == 0 && minnum == 1 && maxnum == 1)
                {
                    // cout<<"code:011"<<endl;
                    return tmpminZ;
                }
                if (divnum == 1 && minnum == 0 && maxnum == 0)
                {
                    // cout<<"code100"<<endl;
                    return Bigmin;
                }
                if (divnum == 0 && minnum == 1 && maxnum == 0)
                {
                    return zvalue;
                    // cout << "BMWRONG!" << endl;
                    // exit(1);
                    // return Bigmin;
                }
                if (divnum == 1 && minnum == 1 && maxnum == 0)
                {
                    return zvalue;
                    // cout << "BMWRONG!" << endl;
                    // exit(1);
                    // return Bigmin;
                }

                if (divnum == 1 && minnum == 0 && maxnum == 1)
                {
                    // cout << "CODE: 101" << endl;
                    // minz = 1000000
                    int innercnt = 0;
                    for (int x0 = 0; x0 < maxlen; x0++)
                    {
                        for (int x1 = 0; x1 < colNum; x1 += 1)
                        {
                            if (x0 >= colbitlength[x1])
                            {
                                continue;
                            }
                            else
                            {
                                if (x0 < i)
                                {
                                    innercnt++;
                                    continue;
                                }
                                else if (x0 == i)
                                {
                                    if (x1 == j)
                                    {
                                        tmpminZ[innercnt] = 1;
                                    }
                                    innercnt += 1;
                                }
                                else
                                {
                                    if (x1 == j)
                                    {
                                        tmpminZ[innercnt] = 0;
                                    }
                                    innercnt += 1;
                                }
                            }
                        }
                    }
                    // Bigmin = tmpminZ;
                    idx += 1;
                    continue;
                }
                if (divnum == 0 && minnum == 0 && maxnum == 1)
                {
                    // cout << "CODE:001" << endl;
                    for (int x00 = 0; x00 < bitlength; x00++)
                    {
                        Bigmin[x00] = tmpminZ[x00];
                    }
                    int innercnt = 0;
                    for (int x0 = 0; x0 < maxlen; x0++)
                    {
                        for (int x1 = 0; x1 < colNum; x1 += 1)
                        {
                            if (x0 >= colbitlength[x1])
                            {
                                continue;
                            }
                            else
                            {
                                if (x0 < i)
                                {
                                    innercnt++;
                                    continue;
                                }
                                else if (x0 == i)
                                {
                                    if (x1 == j)
                                    {
                                        Bigmin[innercnt] = 1;
                                    }
                                    innercnt += 1;
                                }
                                else
                                {
                                    if (x1 == j)
                                    {
                                        Bigmin[innercnt] = 0;
                                    }
                                    innercnt += 1;
                                }
                            }
                        }
                    }
                    innercnt = 0;
                    for (int x0 = 0; x0 < maxlen; x0++)
                    {
                        for (int x1 = 0; x1 < colNum; x1 += 1)
                        {
                            if (x0 >= colbitlength[x1])
                            {
                                continue;
                            }
                            else
                            {
                                if (x0 < i)
                                {
                                    innercnt++;
                                    continue;
                                }
                                else if (x0 == i)
                                {
                                    if (x1 == j)
                                    {
                                        tmpmaxZ[innercnt] = 0;
                                    }
                                    innercnt += 1;
                                }
                                else
                                {
                                    if (x1 == j)
                                    {
                                        tmpmaxZ[innercnt] = 1;
                                    }
                                    innercnt += 1;
                                }
                            }
                        }
                    }
                    // cout << "Tminz:";
                    // for (int i = 0; i < bitlength; i++)
                    // {
                    //     cout << tmpminZ[i];
                    // }
                    // cout << endl;
                    // cout << "Tmaxz:";
                    // for (int i = 0; i < bitlength; i++)
                    // {
                    //     cout << tmpmaxZ[i];
                    // }
                    // cout << endl;
                    // cout << "Bigmi:";
                    // for (int i = 0; i < bitlength; i++)
                    // {
                    //     cout << Bigmin[i];
                    // }
                    // cout << endl;
                    idx += 1;
                    continue;
                }
                idx += 1;
            }
        }
    }
    // cout<<"Normal Ret"<<endl;
    return Bigmin;
}
bool colBinsMin[10][100];
bool colBinsMax[10][100];
bool colBinsVal[10][100];

void incz(bool *value, int len)
{
    if (value[len - 1] == 0)
    {
        value[len - 1] = 1;
        return;
    }
    value[len - 1] = 1;
    for (int i = len - 1; i >= 0; i--)
    {
        if (value[i] == 1)
        {
            value[i] = 0;
        }
        else
        {
            value[i] = 1;
            return;
        }
    }
}

void testTimeRange(MADENet *M, string queryfilepath)
{
    typedef std::chrono::high_resolution_clock Clock;
    long long NonleafCnt = 0;
    long long scanCnt = 0;
    long long loadCnt = 0;
    long long indexTime = 0;
    long long allT = 0;
    ofstream ofs(outputfilepath + "Range.txt");
    Querys *qs = readQueryFile(queryfilepath);
    // Querys *qs = readQueryFile("./data/PowerRangeQuerysSel01.txt");
    // MADENet *M = loadMade("./Model/MadeRoot.txt");
    cout << "Doing range Q" << endl;
    int loopUp = 1;
    for (int loop = 0; loop < loopUp; loop++)
    {
        for (int i = 0; i < qs->queryNumber; i++)
        {
            // i = 466;
            int scanneditem = 0;
            Query qi = qs->Qs[i];
            int *zencode0 = QueryUp2Zvalue(qi, tolbits, 0);
            int *zencode1 = QueryUp2Zvalue(qi, tolbits, 1);
            auto t1 = Clock::now(); // 计时开始
            int *blk0s = pointQueryTriple(M, qi, zencode0);
            // for(int lx =0 ;lx<64;lx+=1){
            //     cout<<zencode0[lx]<<" ";
            // }cout<<endl;
            int *blk1s = pointQueryTriple(M, qi, zencode1);
            auto t1d = Clock::now(); // 计时开始
            NonleafCnt += (std::chrono::duration_cast<std::chrono::nanoseconds>(t1d - t1).count());
            // string first2name0 = "N-" + to_string(blk0s[0]);
            // cout<<"os:"<<blk0s[0]<<" "<<blk0s[1]<<endl;
            // cout<<"os:"<<blk1s[0]<<" "<<blk1s[1]<<endl;
            // continue;

            // int leafIdx0 = 0;
            // if (layer1N[blk0s[0]] == 0)
            // {
            //     for (int ix = blk0s[0] - 1; ix >= 0; ix--)
            //     {
            //         if (layer1N[ix] == 1)
            //         {
            //             blk0s[0] = ix;
            //             first2name0 = "N-" + to_string(blk0s[0]);
            //             break;
            //         }
            //     }
            // }

            // leafIdx0 = leafidx[first2name0];

            // if (layer1N[blk1s[0]] == 0)
            // {
            //     for (int ix = blk1s[0] + 1; ix < M->leafnums; ix++)
            //     {
            //         if (layer1N[ix] == 1)
            //         {
            //             blk1s[0] = ix;

            //             break;
            //         }
            //     }
            // }
            string first2name1 = "N-" + to_string(blk1s[0]);
            // int leafIdx1 = 0;
            // leafIdx1 = leafidx[first2name1];
            int rowcard = 0;
            int uselessBlknum = 0;
            int actScanNum = 0;
            // cout << "starting to search l2" << endl;
            // cout << blk0s[0] << ' ' << blk1s[0] << endl;
            int skipbuk = 0;
            // cout << "ok" << endl;
            for (int leafRidx = blk0s[0]; leafRidx <= blk1s[0]; leafRidx++)
            {
                // string Searchleafname = leafnames[leafRidx];
                string Searchleafname = "N-" + to_string(leafRidx);
                // cout << Searchleafname << endl;
                // CITLeaf *CL = str2memleaf[Searchleafname];
                CITLeaf *CL = readleaf(Searchleafname);
                // readleaf(Searchleafname );
                // cout << "Searching:" << Searchleafname << endl;
                if (CL == NULL)
                {
                    // cout << "Cannot find!" << endl;
                    // cout << Searchleafname << endl;
                    // // exit(1);
                    continue;
                }
                else
                {
                    if (CL->zpageflag == 10)
                    {
                        // cout<<"zp1"<<endl;
                        zpage *lnd = CL->zp;
                        int hasitem = 0;

                        for (int ri = 0; ri < lnd->r; ri++)
                        {
                            int findflag = 1;
                            scanneditem += 1;
                            for (int cj = 0; cj < lnd->c; cj++)
                            {
                                long long value = lnd->data[ri * (lnd->c) + cj];
                                if (value >= qi.leftupBound[cj] && value <= qi.rightdownBound[cj])
                                {
                                    continue;
                                }
                                else
                                {
                                    findflag = 0;
                                    break;
                                }
                            }
                            if (findflag == 1)
                            {
                                // cout<<"Find!"<<Searchleafname + "-" + to_string(subi)<<endl;
                                rowcard += 1;
                                hasitem = 1;
                            }
                        }
                        continue;
                    }

                    int layersubN = 11000;
                    int subidxstart = 0, subidxend = layersubN;
                    if (leafRidx == blk0s[0])
                    {
                        subidxstart = blk0s[1];
                    }
                    if (leafRidx == blk1s[0])
                    {
                        subidxend = blk1s[2];
                    }

                    if (subidxstart == -1)
                    {
                        subidxstart = 0;
                    }
                    if (subidxend == -1)
                    {
                        subidxend = layersubN;
                    }
                    // subidxstart = 1;
                    // cout << "s-e " << subidxstart << " " << subidxend << endl;
                    for (int subi = subidxstart; subi <= subidxend; subi++)
                    {
                        auto tx0 = Clock::now();
                        // CITLeaf *leafx = str2memleaf[Searchleafname + "-" + to_string(subi)];
                        CITLeaf *leafx = readleaf(Searchleafname + "-" + to_string(subi));
                        // cout<<Searchleafname + "-" + to_string(subi)<<endl;
                        auto tx1 = Clock::now();
                        loadCnt += (std::chrono::duration_cast<std::chrono::nanoseconds>(tx1 - tx0).count());
                        if (leafx == NULL)
                        {
                            // cout<<Searchleafname + "-" + to_string(subi)<<"nf"<<endl;
                            break;
                        }
                        else
                        {
                            zpage *lnd = leafx->zp;
                            int hasitem = 0;

                            for (int ri = 0; ri < lnd->r; ri++)
                            {
                                int findflag = 1;
                                scanneditem += 1;
                                for (int cj = 0; cj < lnd->c; cj++)
                                {
                                    long long value = lnd->data[ri * (lnd->c) + cj];
                                    if (value >= qi.leftupBound[cj] && value <= qi.rightdownBound[cj])
                                    {
                                        continue;
                                    }
                                    else
                                    {
                                        findflag = 0;
                                        break;
                                    }
                                }
                                if (findflag == 1)
                                {
                                    // cout<<"Find!"<<Searchleafname + "-" + to_string(subi)<<" ";
                                    rowcard += 1;
                                    hasitem = 1;
                                }
                            }
                            if (hasitem == 0)
                            {
                                // cout<<"hsi0"<<endl;
                                bool minz[128];
                                bool maxz[128];
                                bool ptr[128];
                                for (int copti = 0; copti < lnd->digs - 1; copti++)
                                {
                                    minz[copti] = (bool)zencode0[copti + 1];
                                    maxz[copti] = (bool)zencode1[copti + 1];
                                    ptr[copti] = (bool)lnd->maxz[copti];
                                }
                                // cout<<lnd->digs<<endl;
                                // exit(1);
                                // AnogetBIGMIN(minz, maxz, ptr, lnd->digs - 1, qi.binaryLength, qi.columnNumber);
                                bool *bigmin = getBIGMIN(minz, maxz, ptr, lnd->digs - 1, qi.binaryLength, qi.columnNumber);
                                // cout << "bigmi:";
                                // for (int i = 0; i < 62; i++)
                                // {
                                //     cout << bigmin[i];
                                // }
                                // cout << endl;
                                // for(int i=0;i<32;i++){
                                //     cout<<bigmin[i];
                                // }cout<<endl;
                                // exit(1);
                                int newzipt[128];
                                newzipt[0] = 0;
                                for (int copti = 0; copti < lnd->digs - 1; copti++)
                                {
                                    newzipt[copti + 1] = bigmin[copti];
                                }
                                int *newblk0s = pointQueryTriple(M, qi, newzipt);
                                // cout << "cur state:" << leafRidx << " " << subi << endl;
                                // cout << "Model predict next" << newblk0s[0] << " " << newblk0s[1] << endl;
                                // exit(1);
                                // string namex = "N-" + to_string(newblk0s[0]);
                                // cout<<relaidx<<endl;
                                // cout << "currentStat:" << leafRidx << " " << subi << endl;
                                // cout << "gnz: rela " << relaidx << " act: " << newblk0s[0] << " " << newblk0s[1] << endl;
                                // exit(1);
                                newblk0s[1] -= 1;
                                if (newblk0s[0] == leafRidx)
                                {
                                    if (subi >= newblk0s[1])
                                    {
                                        continue;
                                    }
                                    if (newblk0s[1] != -1)
                                    {
                                        // cout << "successjmp" << endl;
                                        // skipbuk += (newblk0s[1] - subi);
                                        subi = newblk0s[1] - 1;
                                        continue;
                                    }
                                    else
                                    {
                                        continue;
                                    }
                                }
                                // else
                                if (newblk0s[0] != leafRidx)
                                {
                                    if (leafRidx >= newblk0s[0])
                                    {
                                        continue;
                                    }
                                    else

                                    {
                                        // cout << "successjmp" << endl;
                                        leafRidx = newblk0s[0] - 1;
                                        break;
                                    }
                                }
                            }
                        }

                        auto tx2 = Clock::now();
                        // cout<<"scanned time"<<(std::chrono::duration_cast<std::chrono::nanoseconds>(tx2 - tx1).count())<<endl;
                        scanCnt += (std::chrono::duration_cast<std::chrono::nanoseconds>(tx2 - tx1).count());
                    }
                    // cout << "skip: " << skipbuk << " Tolsearch:" << leafIdx1 - leafIdx0 << endl;
                }
            }
            // cout <<"Qid "<<i<< " Rowcard:" << rowcard << " Realcard:" << qid2TrueNumber[i] << " ScannedItem:" << scanneditem << endl;
            auto t2 = Clock::now(); // 计时开始
            allT += (std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count());
            // cout << "Qid:" << i << " Time:" << (std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()) << endl;
            // exit(1);
        }
    }
    ofs << "avg time:(ns)" << allT / (qs->queryNumber * loopUp) << endl;
    ofs << "Noneleaf:" << NonleafCnt / (qs->queryNumber * loopUp) << " Scan:" << scanCnt / (qs->queryNumber * loopUp) << " Load:" << loadCnt / (qs->queryNumber * loopUp) << endl;
    ofs << "Atime:" << NonleafCnt / (qs->queryNumber * loopUp) + loadCnt / (qs->queryNumber * loopUp) + scanCnt / (qs->queryNumber * loopUp) << endl;
    ofs.close();
    cout << "avg time:(ns)" << allT / (qs->queryNumber * loopUp) << endl;
    cout << "Noneleaf:" << NonleafCnt / (qs->queryNumber * loopUp) << " Scan:" << scanCnt / (qs->queryNumber * loopUp) << " Load:" << loadCnt / (qs->queryNumber * loopUp) << endl;
    cout << "Atime:" << NonleafCnt / (qs->queryNumber * loopUp) + loadCnt / (qs->queryNumber * loopUp) + scanCnt / (qs->queryNumber * loopUp) << endl;
}
default_random_engine e;
int randG(float oneProb)
{
    if (u(e) <= oneProb)
    {
        return 1;
    }
    return 0;
}

int lrpermitCheck(long long minv, long long maxv, int encodeLength, int position, long long *rec, int demo)
{
    // cout<<"I'm in"<<endl;
    if (demo == 1)
    {
        cout << "MIN: " << minv << " MAX: " << maxv << " LEN:" << encodeLength << endl;
    }
    //
    bool tmpminv[200];
    bool tmpmaxv[200];
    long long tminv = minv;
    long long tmaxv = maxv;
    for (int i = 0; i < encodeLength; i++)
    {
        tmpminv[encodeLength - i - 1] = tminv % 2;
        tmpmaxv[encodeLength - i - 1] = tmaxv % 2;
        // cout<<tmaxv<<" "<< encodeLength-i<<" "<<tmpmaxv[encodeLength-i-1] <<endl;
        tminv = tminv >> 1;
        tmaxv = tmaxv >> 1;
    }
    if (tmaxv != 0)
    {
        for (int i = 0; i < encodeLength; i++)
        {
            tmpmaxv[encodeLength - i - 1] = 1;
        }
    }
    // cout<<tmaxv<<endl;
    // cout<<"\n";
    if (demo == 1)
    {
        for (int i = 0; i < encodeLength; i++)
        {
            cout << tmpmaxv[i] << " ";
        }
        cout << endl;
        for (int i = 0; i < encodeLength; i++)
        {
            cout << tmpminv[i] << " ";
        }
        cout << endl;
        for (int i = 0; i < position; i++)
        {
            cout << rec[i] << " ";
        }
        cout << endl;
    }
    //
    int leftp = 0; // 0:Nan,1:leftPermit
    int flagx0 = 1;
    for (int i = 0; i < position; i++)
    {
        if (rec[i] > tmpminv[i])
        {
            leftp = 1;
            flagx0 = 0;
            break;
        }
        else if (rec[i] < tmpminv[i])
        {
            leftp = 0;
            flagx0 = 0;
            break;
        }
    }
    if (flagx0 == 1)
    {
        if (tmpminv[position] == 0)
        {
            leftp = 1;
        }
    }
    flagx0 = 1;     // reset
    int rightp = 0; // 0:Nan,1:rightPermit
    for (int i = 0; i < position; i++)
    {
        if (rec[i] > tmpmaxv[i])
        {
            rightp = 0;
            flagx0 = 0;
            break;
        }
        else if (rec[i] < tmpmaxv[i])
        {
            rightp = 1;
            flagx0 = 0;
            break;
        }
    }
    if (flagx0 == 1)
    {
        if (tmpmaxv[position] == 1)
        {
            rightp = 1;
        }
    }
    int retv = 0; // 0:AllNon , 1: Left P,2: rightP,3:AllP
    if (leftp == 0 && rightp == 0)
    {
        retv = 0;
    }
    if (leftp == 1 && rightp == 0)
    {
        retv = 1;
    }
    if (leftp == 0 && rightp == 1)
    {
        retv = 2;
    }
    if (leftp == 1 && rightp == 1)
    {
        retv = 3;
    }

    // cout<<retv<<endl;
    return retv;
}

float drawZ(MADENet *root, Query Qi, int demo)
{
    float out[150];
    float mid[150];
    float p = 1.0;
    int *binaryList = Qi.binaryLength;
    int binaryAllLen = 0;
    int maxBinecn = -1;
    for (int i = 0; i < Qi.columnNumber; i++)
    {
        if (binaryList[i] > maxBinecn)
        {
            maxBinecn = binaryList[i];
        }
        if (binaryList[i] == 0)
        {
            binaryAllLen += 1;
            continue;
        }
        binaryAllLen += binaryList[i];
    }
    int currentdepth = 0;
    long long *searchState[15];
    for (int i = 0; i < Qi.columnNumber; i++)
    {
        int binlen = binaryList[i] + 1;
        searchState[i] = new long long[binlen];
    }
    int flag = 0;
    int samplePoint[150];
    for (int i = 0; i < binaryAllLen; i++)
    {
        samplePoint[i] = 0;
    }
    float OneProbpath[150];
    int innerloopCounter = 0;
    int layer = 0;
    // cout<<maxBinecn<<Qi.columnNumber<<endl;
    for (int i = 0; i < Qi.columnNumber; i++)
    {
        // cout<< binaryList[i]<<" ";
        if (binaryList[i] == 0)
        {
            binaryList[i] = 1;
        }
    }

    for (int i = 0; i < 100; i++)
    {
        samplePoint[i] = 0;
        out[i] = 0;
        mid[i] = 0;
    }
    for (int ix = 0; ix < maxBinecn; ix++)
    {
        for (int j = 0; j < Qi.columnNumber; j++)
        {

            if (ix >= binaryList[j])
            {
                continue;
            }
            else
            {
                // cout<<currentdepth<<endl;
                // currentdepth+=1;
                // continue;
                // eval

                // cout << samplePoint << endl;
                if (demo == 1)
                {
                    cout << ix << " th Dig of col" << j << endl;
                    cout << "Layer:" << layer << endl;
                }
                // cout<<ix<<" th Dig of col"<<j<<endl;
                float OneProb;

                // cout << currentdepth<<endl;
                //  cout<<"curDep"<<currentdepth<<" defaultDepth:"<<defaultDepth<<endl;
                currentdepth += 1;

                MadeIndexInferDig(samplePoint, out, innerloopCounter, innerloopCounter, root, mid);
                OneProb = out[innerloopCounter];
                // cout<<innerloopCounter <<" "<<OneProb<<endl;
                // exit(1);
                OneProbpath[innerloopCounter] = OneProb;
                // if (innerloopCounter > 80)
                // {
                //     return p;
                // }
                if (demo == 1)
                {
                    cout << "ilc" << innerloopCounter << " NetInput:" << samplePoint[innerloopCounter] << endl;
                    cout << "OneProb: " << OneProb << " P: " << p << endl;
                    // if (innerloopCounter ==3){
                    //     exit(1);
                    // }
                }
                // cout<<"OneProb: "<<OneProb<<" P: "<<p<<endl;
                // 根据查询剪枝
                int ff = lrpermitCheck(Qi.leftupBound[j], Qi.rightdownBound[j], Qi.binaryLength[j], ix, searchState[j], demo);
                if (demo == 1)
                {
                    cout << "PermitState:" << ff << endl;
                }
                //

                if (ff == 0)
                {
                    p = 0;
                    cout << "p0" << endl;
                    exit(1);
                    // return 0;
                }
                if (ff == 1)
                {
                    samplePoint[innerloopCounter] = 0;
                    searchState[j][ix] = 0;
                    p = p * (1 - OneProb);
                }
                if (ff == 2)
                {
                    samplePoint[innerloopCounter] = 1;
                    searchState[j][ix] = 1;
                    p = p * OneProb;
                }
                if (ff == 3)
                { // 生成采样点
                    samplePoint[innerloopCounter] = randG(OneProb);
                    searchState[j][ix] = samplePoint[innerloopCounter];
                }
                innerloopCounter += 1;
            }
        }
    }
    // cout<<"est done"<<endl;
    // for (int i = 0; i < 20; i++)
    // {
    //     cout << samplePoint[i] << " ";
    // }
    // for (int i = 0; i < 20; i++)
    // {
    //     cout << out[i] << " ";
    // }

    // cout<<"Over"<<endl;
    // cout<<p<<endl;
    // exit(1) ;
    for (int i = 0; i < Qi.columnNumber; i++)
    {
        delete searchState[i];
    }
    return p;

    return 0.0;
}
int cardEstimate(MADENet *root, Query Qi, int sampleNumber)
{
    float p = 0;
    int demo = 0;
    for (int i = 0; i < sampleNumber; i++)
    {
        // if (i==1){
        //     demo=1;

        // }
        // cout<<i<<endl;
        p += drawZ(root, Qi, demo);
        if (demo == 1)
        {
            cout << p << endl;
            exit(1);
        }
        /* code */
    }
    p = p / sampleNumber;
    return int(root->zdr * (p));
}
void ZADD(int *zencode0, int *zencode1, int *ans, int len)
{
    int next = 0;
    // for(int i=0;i<len;i++){
    //     cout<<zencode0[i];
    // }cout<<endl;
    // for(int i=0;i<len;i++){
    //     cout<<zencode1[i];
    // }cout<<endl;

    for (int i = len - 1; i >= 0; i--)
    {
        int val = zencode0[i] + zencode1[i] + next;
        if (val == 2)
        {
            next = 1;
            ans[i] = 0;
        }
        else if (val == 1)
        {
            next = 0;
            ans[i] = 1;
        }
        else if (val == 3)
        {
            next = 1;
            ans[i] = 1;
        }
        else
        {
            next = 0;
            ans[i] = 0;
        }
    }
    // for(int i=0;i<len;i++){
    //     cout<<ans[i];
    // }cout<<endl;
}
bool inbox(Query Qi, int *zencode)
{
    int maxlen = 0;
    long long zdecode[20];
    for (int i = 0; i < Qi.columnNumber; i += 1)
    {
        zdecode[i] = 0;
        if (Qi.binaryLength[i] > maxlen)
        {
            maxlen = Qi.binaryLength[i];
        }
    }
    int idx = 0;
    for (int i = 0; i < maxlen; i += 1)
    {
        for (int j = 0; j < Qi.columnNumber; j++)
        {
            if (i >= Qi.binaryLength[j])
            {
                continue;
            }
            else
            {
                zdecode[j] *= 2;
                zdecode[j] += zencode[idx];
                idx += 1;
            }
        }
    }
    for (int i = 0; i < Qi.columnNumber; i++)
    {
        if (zdecode[i] < Qi.leftupBound[i] || zencode[i] > Qi.rightdownBound[i])
        {
            return false;
        }
    }
    return true;
}
float probeCDF(int *zencode0, int *zencode1, int diglen, MADENet *M, Query Qi, int depth)
{
    int *mid = new int[diglen + 10];
    float cdfl = point2cdfest(M, Qi, zencode0);
    float cdfu = point2cdfest(M, Qi, zencode1);

    if (depth == 0)
    {
        return cdfu - cdfl;
    }
    if ((cdfu - cdfl) < 0.00001)
    {
        return cdfu - cdfl;
    }
    if (cdfl > cdfu)
    {
        return 0;
    }

    ZADD(zencode0, zencode1, &mid[1], diglen);
    // cout<<"MIN:";
    // for(int i=0;i<32;i++){
    //     cout<<zencode0[i+1];
    // }cout<<endl;
    // cout<<"MAX:";
    // for(int i=0;i<32;i++){
    //     cout<<zencode1[i+1];
    // }cout<<endl;
    // cout<<"MID:";
    // for(int i=0;i<32;i++){
    //     cout<<mid[i+1];
    // }cout<<endl;
    mid[0] = 0;
    float cdfm = point2cdfest(M, Qi, mid);
    // cout <<"Dep"<<depth<<" L:"<< cdfl << " R:" << cdfu <<" mid:" <<cdfm<<endl;
    bool flag = inbox(Qi, mid);
    if (flag == true)
    {
        // cout<<"inbox"<<endl;
        int lm[120];
        int mu[120];
        lm[0] = 0;
        mu[0] = 0;
        ZADD(zencode0, mid, &lm[1], diglen);
        ZADD(zencode1, mid, &mu[1], diglen);
        float l = probeCDF(zencode0, lm, diglen, M, Qi, depth - 1);
        float u = probeCDF(lm, zencode1, diglen, M, Qi, depth - 1);
        return l + u;
    }
    else
    {
        // cout<<"ob"<<endl;
        bool minb[120];
        bool maxb[120];
        bool midb[120];
        int tmpbmi[120];
        int tmpmai[120];
        for (int i = 0; i < diglen; i++)
        {
            minb[i] = zencode0[i + 1];
            maxb[i] = zencode1[i + 1];
            midb[i] = mid[i + 1];
        }
        bool *bigmin = getBIGMIN(minb, maxb, midb, diglen, Qi.binaryLength, Qi.columnNumber);
        bool *litmax = getLITMAX(minb, maxb, midb, diglen, Qi.binaryLength, Qi.columnNumber);
        for (int i = 0; i < diglen; i++)
        {

            tmpbmi[i + 1] = bigmin[i];
            tmpmai[i + 1] = litmax[i];
        }
        tmpbmi[0] = 0;
        tmpmai[0] = 0;
        // cout<<cdfl<<" "<<point2cdfest(M,Qi,tmpmai)<<" "<<point2cdfest(M,Qi,tmpbmi)<<" "<<cdfu<<endl;
        float ret = 0;
        ret += probeCDF(zencode0, tmpmai, diglen, M, Qi, depth - 1);
        ret += probeCDF(tmpbmi, zencode1, diglen, M, Qi, depth - 1);
        // cout<<abs( cdfl - point2cdfest(M,Qi,tmpmai)  )<<endl;
        // if(abs( cdfl - point2cdfest(M,Qi,tmpmai)  )>= 0.000001){
        //     ret += probeCDF(zencode0,tmpbmi,diglen,M,Qi,depth-1);
        // }
        // if(abs(cdfu -point2cdfest(M,Qi,tmpbmi)) >=0.000001){
        //     ret+= probeCDF(tmpmai,zencode1,diglen,M,Qi,depth-1);
        // }

        return ret;
        // exit(1);
    }
    // cout << cdfm << endl;
}
void testCardPerformance(MADENet *M, string queryfilepath)
{
    vector<float> pdist;
    // Querys* qs = readQueryFile("./data/osmQuerys.txt");
    Querys *qs = readQueryFile(queryfilepath);
    cout << "Query loaded" << endl;
    ofstream ofs(outputfilepath + "Card.txt");
    // MADENet *M = loadMade("./Model/MadeRoot.txt");
    int sampleN = 2000;
    float p50, p95, p90, p99;
    vector<float> ABSL;
    typedef std::chrono::high_resolution_clock Clock;
    long long timesum = 0, NonleafCnt = 0;
    long long queryestTime = 0;
    for (int i = 0; i < qs->queryNumber; i++)
    {
        // i = 25;
        Query qi = qs->Qs[i];
        // cout<<"QID"<<i<<endl;
        auto queryfirst = Clock::now(); // 计时开始
        int realcard, estcard;
        realcard = qid2TrueNumber[i];
        int *zencode0 = QueryUp2Zvalue(qi, tolbits, 0);
        auto t1x = Clock::now(); // 计时开始
        int *blk0s = pointQueryTriple(M, qi, zencode0);
        auto t1d = Clock::now(); // 计时开始
        int *zencode1 = QueryUp2Zvalue(qi, tolbits, 1);
        auto t2x = Clock::now(); // 计时开始
        int *blk1s = pointQueryTriple(M, qi, zencode1);
        auto t2d = Clock::now(); // 计时开始
        // cout << "Start2 update method" << endl;
        float f = probeCDF(zencode0, zencode1, *tolbits, M, qi,16);
        // f=1;
        // cout<<"fi"<<f<<endl;
        // exit(1);
        // NonleafCnt += (std::chrono::duration_cast<std::chrono::nanoseconds>(t1d - t1x).count());
        // NonleafCnt += (std::chrono::duration_cast<std::chrono::nanoseconds>(t2d - t2x).count());
        // cout << blk0s[0] << " " << blk1s[0] << endl;
        // if (false)
        f=10000;
        float RoughEst = f * M->zdr;
        
        if (RoughEst <= 0.01 * M->zdr)
        {
            // cout << "same blk" << endl;

            string first2name0 = "N-" + to_string(blk0s[0]);
            int leafIdx0 = 0;
            if (layer1N[blk0s[0]] == 0)
            {
                for (int ix = blk0s[0] - 1; ix >= 0; ix--)
                {
                    if (layer1N[ix] == 1)
                    {
                        blk0s[0] = ix;
                        break;
                    }
                }
            }
            leafIdx0 = leafidx[first2name0];
            if (leafIdx0 == -1)
            {
                for (int newi = blk0s[0] - 1; newi >= 0; newi--)
                {
                    first2name0 = "N-" + to_string(newi);
                    if (leafidx[first2name0] != -1)
                    {
                        blk0s[0] = newi;
                        leafIdx0 = blk0s[0];
                        blk0s[1] = 0;
                        break;
                    }
                }
            }
            int *zencode1 = QueryUp2Zvalue(qi, tolbits, 1);
            int *blk1s = pointQueryTriple(M, qi, zencode1);
            if (layer1N[blk1s[0]] == 0)
            {
                for (int ix = blk1s[0] + 1; ix < M->leafnums; ix++)
                {
                    if (layer1N[ix] == 1)
                    {
                        blk1s[0] = ix;
                        break;
                    }
                }
            }
            string first2name1 = "N-" + to_string(blk1s[0]);
            int leafIdx1 = 0;
            leafIdx1 = leafidx[first2name1];
            if (leafIdx1 == -1)
            {
                for (int newi = blk0s[1]; newi < M->leafnums; newi++)
                {
                    first2name1 = "N-" + to_string(newi);
                    if (leafidx[first2name0] != -1)
                    {
                        blk1s[0] = newi;
                        leafIdx0 = blk1s[0];
                        blk1s[1] = 0;
                        break;
                    }
                }
            }
            int rowcard = 0;
            int uselessBlknum = 0;
            int actScanNum = 0;
            // cout << "starting to search l2" << endl;
            // cout << leafIdx0 << ' ' << leafIdx1 << endl;
            int skipbuk = 0;
            // cout << "ok" << endl;
            for (int leafRidx = leafIdx0; leafRidx <= leafIdx1; leafRidx++)
            {
                string Searchleafname = leafnames[leafRidx];
                // cout << Searchleafname << endl;

                CITLeaf *CL = str2memleaf[Searchleafname];
                // readleaf(Searchleafname );
                // cout << "Searching:" << Searchleafname << endl;
                if (CL == NULL)
                {
                    // cout << "Cannot find!" << endl;
                    // cout << Searchleafname << endl;
                    // // exit(1);
                    continue;
                }
                else
                {

                    int layersubN = 11000;
                    int subidxstart = 0, subidxend = layersubN;
                    if (leafRidx == leafIdx0)
                    {
                        subidxstart = blk0s[1];
                    }
                    if (leafRidx == leafIdx1)
                    {
                        subidxend = blk1s[1];
                    }

                    if (subidxstart == -1)
                    {
                        subidxstart = 0;
                    }
                    if (subidxend == -1)
                    {
                        subidxend = layersubN;
                    }
                    // subidxstart = 1;
                    // cout << "s-e " << subidxstart << " " << subidxend << endl;
                    for (int subi = subidxstart; subi <= subidxend; subi++)
                    {
                        auto tx0 = Clock::now();
                        CITLeaf *leafx = str2memleaf[Searchleafname + "-" + to_string(subi)];
                        // readleaf(Searchleafname + "-" + to_string(subi));
                        auto tx1 = Clock::now();
                        if (leafx == NULL)
                        {
                            break;
                        }
                        else
                        {
                            zpage *lnd = leafx->zp;
                            int hasitem = 0;

                            for (int ri = 0; ri < lnd->r; ri++)
                            {
                                int findflag = 1;
                                for (int cj = 0; cj < lnd->c; cj++)
                                {
                                    long long value = lnd->data[ri * (lnd->c) + cj];
                                    if (value >= qi.leftupBound[cj] && value <= qi.rightdownBound[cj])
                                    {
                                        continue;
                                    }
                                    else
                                    {
                                        findflag = 0;
                                        break;
                                    }
                                }
                                if (findflag == 1)
                                {
                                    // cout<<"Find!"<<Searchleafname + "-" + to_string(subi)<<endl;
                                    rowcard += 1;
                                    hasitem = 1;
                                }
                            }
                            if (hasitem == 0)
                            {
                                bool *minz = new bool[128];
                                bool *maxz = new bool[128];
                                bool *ptr = new bool[128];
                                for (int copti = 0; copti < lnd->digs - 1; copti++)
                                {
                                    minz[copti] = zencode0[copti + 1];
                                    maxz[copti] = zencode1[copti + 1];
                                    ptr[copti] = lnd->maxz[copti];
                                }
                                bool *bigmin = getBIGMIN(minz, maxz, ptr, lnd->digs - 1, qi.binaryLength, qi.columnNumber);
                                int newzipt[128];
                                newzipt[0] = 0;
                                for (int copti = 0; copti < lnd->digs - 1; copti++)
                                {
                                    newzipt[copti + 1] = bigmin[copti];
                                }
                                int *newblk0s = pointQueryTriple(M, qi, newzipt);
                                // newblk0s[1]-=100;
                                // cout<<"Model predict next"<<newblk0s[0]<<" "<<newblk0s[1]<<endl;
                                int relaidx = leafidx["N-" + to_string(newblk0s[0])];
                                if (relaidx == -1)
                                {
                                    for (int relai = newblk0s[0]; relai < M->leafnums; relai++)
                                    {
                                        if (leafidx["N-" + to_string(relai)] != -1)
                                        {
                                            relaidx = leafidx["N-" + to_string(relai)];
                                            newblk0s[0] = relai;
                                            newblk0s[1] = 0;
                                            break;
                                        }
                                    }
                                }
                                // cout << "currentStat:" << leafRidx << " " << subi << endl;
                                // cout << "gnz: rela " << relaidx << " act: " << newblk0s[0] << " " << newblk0s[1] << endl;
                                // exit(1);

                                if (relaidx == leafRidx)
                                {
                                    if (subi >= newblk0s[1])
                                    {
                                        continue;
                                    }
                                    if (newblk0s[1] != -1)
                                    {
                                        skipbuk += (newblk0s[1] - subi);
                                        subi = newblk0s[1] - 1;
                                        continue;
                                    }
                                    else
                                    {
                                        leafRidx = leafRidx + 1;
                                        break;
                                    }
                                }
                                else
                                {
                                    if (leafRidx >= relaidx)
                                    {
                                        break;
                                    }
                                    else
                                    {
                                        leafRidx = relaidx - 1;
                                        break;
                                    }
                                }
                            }
                        }

                        auto tx2 = Clock::now();
                        // cout<<"scanned time"<<(std::chrono::duration_cast<std::chrono::nanoseconds>(tx2 - tx1).count())<<endl;
                    }
                    // cout << "skip: " << skipbuk << " Tolsearch:" << leafIdx1 - leafIdx0 << endl;
                }
            }
            // exit(1);
        }
        // // cout<<blknum0<<" "<<blknum1<<endl;
        // if ((blknum1 - blknum0) <= 1)
        // {
        //     estcard = 0;
        //     for (int blkidx = blknum0; blkidx <= blknum1; blkidx++)
        //     {
        //         leafDec *lnd = readDec(to_string(blkidx) + "d");
        //         if (lnd == NULL)
        //         {
        //             continue;
        //         }
        //         // cout << "Loading takes :" << (std::chrono::duration_cast<std::chrono::nanoseconds>(t6 - t5).count()) << endl;
        //         if (lnd->r == 0)
        //         {
        //             continue;
        //         }
        //         else
        //         {

        //             for (int ri = 0; ri < lnd->r; ri++)
        //             {
        //                 int findflag = 1;
        //                 for (int cj = 0; cj < lnd->c; cj++)
        //                 {
        //                     long long value = lnd->data[ri * (lnd->c) + cj];

        //                     if (value >= qi.leftupBound[cj] && value <= qi.rightdownBound[cj])
        //                     {
        //                         continue;
        //                     }
        //                     else
        //                     {
        //                         findflag = 0;
        //                         break;
        //                     }
        //                 }
        //                 if (findflag == 1)
        //                 {
        //                     estcard += 1;
        //                 }
        //             }
        //         }
        //     }
        // }
        else
        {
            // cout << "CE" << endl;
            auto t3 = Clock::now(); // 计时开始
            estcard = cardEstimate(M, qi, sampleN);
            auto t3d = Clock::now(); // 计时开始
            long delta = (std::chrono::duration_cast<std::chrono::nanoseconds>(t3d - t3).count());
            cout << "Card Est time:" << delta << endl;
            // cout << estcard << endl;
            timesum += (std::chrono::duration_cast<std::chrono::nanoseconds>(t3d - t3).count());
        }
        auto queryend = Clock::now(); // 计时开始
        cout << "Qid" << qi.queryid << " CDF Approx:" << f * M->zdr << "\tRealCard:" << realcard << "\tEstCard:" << estcard << " P :" << pErrorcalculate(estcard, realcard) << endl;
        queryestTime += (std::chrono::duration_cast<std::chrono::nanoseconds>(queryend - queryfirst).count());
        cout << "Query est time:" << to_string(std::chrono::duration_cast<std::chrono::nanoseconds>(queryend - queryfirst).count()) << "ns" << endl;
        ofs    << pErrorcalculate(estcard, realcard) << endl;
        
        // ofs << "Qid" << qi.queryid << "\tRealCard:" << realcard << "\tEstCard:" << estcard << " P :" << pErrorcalculate(estcard, realcard) << endl;
        pdist.push_back(pErrorcalculate(estcard, realcard));
        // exit(1);
    }
    cout << timesum / qs->queryNumber << endl;
    sort(pdist.begin(), pdist.end());
    p50 = pdist[((int)(0.5 * pdist.size()))];
    p90 = pdist[((int)(0.95 * pdist.size()))];
    p95 = pdist[((int)(0.99 * pdist.size()))];
    // p99 = pdist[((int)(0.99 * pdist.size()))];
    float pmax = pdist[pdist.size() - 1];
    cout << "P50\tP90\tP95\tPmax\tAvgT" << endl;
    cout << p50 << "\t" << p90 << '\t' << p95 << '\t' << pmax << '\t' << queryestTime / qs->queryNumber << endl;
    ofs << "P50\tP90\tP95\tPmax\tAvgT" << endl;
    ofs << p50 << "\t" << p90 << '\t' << p95 << '\t' << pmax << '\t' << queryestTime / qs->queryNumber << endl;
    ofs.close();
}
// int *input = new int[20];
// float *mid = new float[20];
// float *out = new float[20];
// void testGNZ()
// {
//     bool zv[] =   {0, 0, 1, 1, 1, 0, 1, 0};
//     bool minz[] = {0, 0, 0, 1, 1, 0, 1, 1};
//     bool maxz[] = {0, 1, 1, 0, 0, 1, 1, 0};

//     int bl[] = {4, 4};

//     AnogetBIGMIN(minz, maxz, zv, 8, bl, 2);
//     for (int i = 0; i < 8; i++)
//     {
//         cout << zv[i] << "";
//     }
//     cout << endl;
// }
void testPoiRangeX(MADENet *M, string queryfilepath)
{
    typedef std::chrono::high_resolution_clock Clock;
    long long NonleafCnt = 0;
    long long scanCnt = 0;
    long long loadCnt = 0;
    long long indexTime = 0;
    long long allT = 0;
    ofstream ofs(outputfilepath + "PointQ.txt");
    Querys *qs = readQueryFile(queryfilepath);
    // Querys *qs = readQueryFile("./data/PowerRangeQuerysSel01.txt");
    // MADENet *M = loadMade("./Model/MadeRoot.txt");
    cout << "Doing P Q" << endl;
    int loopUp = 1000;
    for (int loop = 0; loop < loopUp; loop++)
    {
        for (int i = 0; i < qs->queryNumber; i++)
        {
            // i = 466;
            int scanneditem = 0;
            Query qi = qs->Qs[i];
            int *zencode0 = QueryUp2Zvalue(qi, tolbits, 0);
            int *zencode1 = QueryUp2Zvalue(qi, tolbits, 0);
            auto t1 = Clock::now(); // 计时开始
            int *blk0s = pointQueryTriple(M, qi, zencode0);
            // for(int lx =0 ;lx<64;lx+=1){
            //     cout<<zencode0[lx]<<" ";
            // }cout<<endl;
            // cout<<"os:"<<blk0s[0]<<" "<<blk0s[1]<<endl;
            auto t1d = Clock::now(); // 计时开始
            NonleafCnt += (std::chrono::duration_cast<std::chrono::nanoseconds>(t1d - t1).count());
            continue;
            int *blk1s = pointQueryTriple(M, qi, zencode1);

            // string first2name0 = "N-" + to_string(blk0s[0]);
            // cout<<"os:"<<blk0s[0]<<" "<<blk0s[1]<<endl;
            // cout<<"os:"<<blk1s[0]<<" "<<blk1s[1]<<endl;
            // continue;

            // int leafIdx0 = 0;
            // if (layer1N[blk0s[0]] == 0)
            // {
            //     for (int ix = blk0s[0] - 1; ix >= 0; ix--)
            //     {
            //         if (layer1N[ix] == 1)
            //         {
            //             blk0s[0] = ix;
            //             first2name0 = "N-" + to_string(blk0s[0]);
            //             break;
            //         }
            //     }
            // }

            // leafIdx0 = leafidx[first2name0];

            // if (layer1N[blk1s[0]] == 0)
            // {
            //     for (int ix = blk1s[0] + 1; ix < M->leafnums; ix++)
            //     {
            //         if (layer1N[ix] == 1)
            //         {
            //             blk1s[0] = ix;

            //             break;
            //         }
            //     }
            // }
            string first2name1 = "N-" + to_string(blk1s[0]);
            // int leafIdx1 = 0;
            // leafIdx1 = leafidx[first2name1];
            int rowcard = 0;
            int uselessBlknum = 0;
            int actScanNum = 0;
            // cout << "starting to search l2" << endl;
            // cout << blk0s[0] << ' ' << blk1s[0] << endl;
            int skipbuk = 0;
            // cout << "ok" << endl;
            for (int leafRidx = blk0s[0]; leafRidx <= blk1s[0]; leafRidx++)
            {
                // string Searchleafname = leafnames[leafRidx];
                string Searchleafname = "N-" + to_string(leafRidx);
                // cout << Searchleafname << endl;
                // CITLeaf *CL = str2memleaf[Searchleafname];
                CITLeaf *CL = readleaf(Searchleafname);
                // readleaf(Searchleafname );
                // cout << "Searching:" << Searchleafname << endl;
                if (CL == NULL)
                {
                    // cout << "Cannot find!" << endl;
                    // cout << Searchleafname << endl;
                    // // exit(1);
                    continue;
                }
                else
                {
                    if (CL->zpageflag == 10)
                    {
                        // cout<<"zp1"<<endl;
                        zpage *lnd = CL->zp;
                        int hasitem = 0;

                        for (int ri = 0; ri < lnd->r; ri++)
                        {
                            int findflag = 1;
                            scanneditem += 1;
                            for (int cj = 0; cj < lnd->c; cj++)
                            {
                                long long value = lnd->data[ri * (lnd->c) + cj];
                                if (value >= qi.leftupBound[cj] && value <= qi.rightdownBound[cj])
                                {
                                    continue;
                                }
                                else
                                {
                                    findflag = 0;
                                    break;
                                }
                            }
                            if (findflag == 1)
                            {
                                // cout<<"Find!"<<Searchleafname + "-" + to_string(subi)<<endl;
                                rowcard += 1;
                                hasitem = 1;
                            }
                        }
                        continue;
                    }

                    int layersubN = 11000;
                    int subidxstart = 0, subidxend = layersubN;
                    if (leafRidx == blk0s[0])
                    {
                        subidxstart = blk0s[1];
                    }
                    if (leafRidx == blk1s[0])
                    {
                        subidxend = blk1s[2];
                    }

                    if (subidxstart == -1)
                    {
                        subidxstart = 0;
                    }
                    if (subidxend == -1)
                    {
                        subidxend = layersubN;
                    }
                    // subidxstart = 1;
                    // cout << "s-e " << subidxstart << " " << subidxend << endl;
                    for (int subi = subidxstart; subi <= subidxend; subi++)
                    {
                        auto tx0 = Clock::now();
                        // CITLeaf *leafx = str2memleaf[Searchleafname + "-" + to_string(subi)];
                        CITLeaf *leafx = readleaf(Searchleafname + "-" + to_string(subi));
                        // cout<<Searchleafname + "-" + to_string(subi)<<endl;
                        auto tx1 = Clock::now();
                        loadCnt += (std::chrono::duration_cast<std::chrono::nanoseconds>(tx1 - tx0).count());
                        if (leafx == NULL)
                        {
                            // cout<<Searchleafname + "-" + to_string(subi)<<"nf"<<endl;
                            break;
                        }
                        else
                        {
                            zpage *lnd = leafx->zp;
                            int hasitem = 0;

                            for (int ri = 0; ri < lnd->r; ri++)
                            {
                                int findflag = 1;
                                scanneditem += 1;
                                for (int cj = 0; cj < lnd->c; cj++)
                                {
                                    long long value = lnd->data[ri * (lnd->c) + cj];
                                    if (value >= qi.leftupBound[cj] && value <= qi.rightdownBound[cj])
                                    {
                                        continue;
                                    }
                                    else
                                    {
                                        findflag = 0;
                                        break;
                                    }
                                }
                                if (findflag == 1)
                                {
                                    // cout<<"Find!"<<Searchleafname + "-" + to_string(subi)<<" ";
                                    rowcard += 1;
                                    hasitem = 1;
                                }
                            }
                            if (hasitem == 0)
                            {
                                // cout<<"hsi0"<<endl;
                                bool minz[128];
                                bool maxz[128];
                                bool ptr[128];
                                for (int copti = 0; copti < lnd->digs - 1; copti++)
                                {
                                    minz[copti] = (bool)zencode0[copti + 1];
                                    maxz[copti] = (bool)zencode1[copti + 1];
                                    ptr[copti] = (bool)lnd->maxz[copti];
                                }
                                // cout<<lnd->digs<<endl;
                                // exit(1);
                                // AnogetBIGMIN(minz, maxz, ptr, lnd->digs - 1, qi.binaryLength, qi.columnNumber);
                                bool *bigmin = getBIGMIN(minz, maxz, ptr, lnd->digs - 1, qi.binaryLength, qi.columnNumber);
                                // cout << "bigmi:";
                                // for (int i = 0; i < 62; i++)
                                // {
                                //     cout << bigmin[i];
                                // }
                                // cout << endl;
                                // for(int i=0;i<32;i++){
                                //     cout<<bigmin[i];
                                // }cout<<endl;
                                // exit(1);
                                int newzipt[128];
                                newzipt[0] = 0;
                                for (int copti = 0; copti < lnd->digs - 1; copti++)
                                {
                                    newzipt[copti + 1] = bigmin[copti];
                                }
                                int *newblk0s = pointQueryTriple(M, qi, newzipt);
                                // cout << "cur state:" << leafRidx << " " << subi << endl;
                                // cout << "Model predict next" << newblk0s[0] << " " << newblk0s[1] << endl;
                                // exit(1);
                                // string namex = "N-" + to_string(newblk0s[0]);
                                // cout<<relaidx<<endl;
                                // cout << "currentStat:" << leafRidx << " " << subi << endl;
                                // cout << "gnz: rela " << relaidx << " act: " << newblk0s[0] << " " << newblk0s[1] << endl;
                                // exit(1);
                                newblk0s[1] -= 1;
                                if (newblk0s[0] == leafRidx)
                                {
                                    if (subi >= newblk0s[1])
                                    {
                                        continue;
                                    }
                                    if (newblk0s[1] != -1)
                                    {
                                        // cout << "successjmp" << endl;
                                        // skipbuk += (newblk0s[1] - subi);
                                        subi = newblk0s[1] - 1;
                                        continue;
                                    }
                                    else
                                    {
                                        continue;
                                    }
                                }
                                // else
                                if (newblk0s[0] != leafRidx)
                                {
                                    if (leafRidx >= newblk0s[0])
                                    {
                                        continue;
                                    }
                                    else

                                    {
                                        // cout << "successjmp" << endl;
                                        leafRidx = newblk0s[0] - 1;
                                        break;
                                    }
                                }
                            }
                        }

                        auto tx2 = Clock::now();
                        // cout<<"scanned time"<<(std::chrono::duration_cast<std::chrono::nanoseconds>(tx2 - tx1).count())<<endl;
                        scanCnt += (std::chrono::duration_cast<std::chrono::nanoseconds>(tx2 - tx1).count());
                    }
                    // cout << "skip: " << skipbuk << " Tolsearch:" << leafIdx1 - leafIdx0 << endl;
                }
            }
            // cout <<"Qid "<<i<< " Rowcard:" << rowcard << " Realcard:" << qid2TrueNumber[i] << " ScannedItem:" << scanneditem << endl;
            auto t2 = Clock::now(); // 计时开始
            allT += (std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count());
            // cout << "Qid:" << i << " Time:" << (std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count()) << endl;
            // exit(1);
        }
    }
    ofs << "avg time:(ns)" << allT / (qs->queryNumber * loopUp) << endl;
    ofs << "Noneleaf:" << NonleafCnt / (qs->queryNumber * loopUp) << " Scan:" << scanCnt / (qs->queryNumber * loopUp) << " Load:" << loadCnt / (qs->queryNumber * loopUp) << endl;
    ofs << "Atime:" << NonleafCnt / (qs->queryNumber * loopUp) + loadCnt / (qs->queryNumber * loopUp) + scanCnt / (qs->queryNumber * loopUp) << endl;
    ofs.close();
    cout << "avg time:(ns)" << allT / (qs->queryNumber * loopUp) << endl;
    cout << "Noneleaf:" << NonleafCnt / (qs->queryNumber * loopUp) << " Scan:" << scanCnt / (qs->queryNumber * loopUp) << " Load:" << loadCnt / (qs->queryNumber * loopUp) << endl;
    cout << "Atime:" << NonleafCnt / (qs->queryNumber * loopUp) + loadCnt / (qs->queryNumber * loopUp) + scanCnt / (qs->queryNumber * loopUp) << endl;
}

int main(int argc, char *argv[])
{
    // testGNZ();
    // exit(1);

    string queryfilepath = (argv[1]);
    string jobname = argv[2];
    outputfilepath = argv[3];
    leafSubname = argv[4];
    // string
    cout << "Working on " << jobname << " with query: " << queryfilepath << endl;

    MADENet *M = loadMade("./Model/MadeRoot" + leafSubname);
    cout << "Index Size: CORE:" << rootCoreSize / 1024 << "KB Nonleaf:" << nonleafSize / (1024 * 1024) << "MB "<<nonleafSize / (1024) << "KB" << endl;
    ofstream ofs("./result/CITSizeInfo-" + leafSubname);
    ofs << "Index Size: CORE:" << rootCoreSize / 1024 << "KB Nonleaf:" << nonleafSize / (1024 * 1024) << "MB "<<nonleafSize / (1024)<< "KB"  << endl;
    ofs.close();
    
    if (jobname == "CARD")
    {
        testCardPerformance(M, queryfilepath);
        // testTimeRange(M, queryfilepath);
        // testTimePointQ(M, queryfilepath);
        // testPoiRangeX(M,queryfilepath);
    }
    else{
        cout << "Range Ready" << endl;
                // testCardPerformance(M, queryfilepath);
        // testTimeRange(M, queryfilepath);
        cout<<"Point Q:"<<endl;
        testPoiRangeX(M,queryfilepath);
        // testTimePointQ(M, queryfilepath);
    } // if (jobname == "CARD")
      // {

    // // }
    // else if (jobname == "RI")
    // {
    //     testTimeRange(queryfilepath);
    //     /* code */
    // }

    // for (int i = 0; i < 20; i++)
    // {
    //     input[i] = 0;
    //     mid[i] = 0;
    //     out[i] = 0;
    // }
    // MADENet *M = loadMade("./Model/MadeRoot.txt");
    // for (int i = 0; i < 10; i++)
    // {
    //     MadeIndexInferDig(input, out, i, i, M, mid);
    // }
    // for (int i = 0; i < 10; i++)
    // {
    //     cout << out[i] << " ";
    // }

    // testTimeRange();
    // testTimePointQ();
}
