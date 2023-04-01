

import sys

if __name__ =="__main__":
    # print(sys.argc
    nparam = len(sys.argv)
    print(nparam)
    mergeout = sys.argv[1]
    # print(mergeout)
    # exit(1)
    fout = open(mergeout,'w')
    for i in range(nparam-2):
        inname = sys.argv[2+i]
        print('writing :' ,inname)
        fin = open(inname,'r')
        fout.write('---------------------\n')
        fout.write(inname)
        fout.write('\n')
        lines = fin.readlines()
        for l in lines:
            fout.write(l)
            fout.write('\n')
        fout.write('\n')
        fin.close()
    fout.close()
