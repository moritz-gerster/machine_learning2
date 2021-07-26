def benchmark(text2sow,kernel):

    # Extract the two longest texts in the dataset
    texts = list(open('newsgroup-data/newsgroup-train-data.txt','r'))
    sortedtexts = sorted(texts,key=len)[::-1]

    longtext1 = sortedtexts[0]
    longtext2 = sortedtexts[1]

    # Build their set-of-words representation
    sow1 = text2sow(longtext1)
    sow2 = text2sow(longtext2)

    # Apply the kernel function and compute its output and running time
    import time
    start = time.time(); output = kernel(sow1,sow2); end = time.time()
    print('kernel score: %.3f , computation time: %.3f'%(output,end-start))

def naivekernel(sow1,sow2):
    k = 0
    L = sow1|sow2
    for w in L:

        insow1 = False
        for v in sow1:
            if v==w: insow1 = True

        insow2 = False
        for v in sow2:
            if v==w: insow2 = True

        if insow1 and insow2: k+=1

    return k

