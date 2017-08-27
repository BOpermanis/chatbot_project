from multiprocessing import Pool

def g(a,b):
    return a*b

def f(x):
    print(x)
    return g(x,x)

if __name__ == '__main__':
    with Pool(4) as p:
        print(p.map(f, range(100)))