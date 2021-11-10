import random

def rando(n):
    a = []
    b = []
    for i in range(n):
        a.append(random.randint(0,20))
        b.append(random.randint(0,20))

    b.sort()
    a.sort()
    print(a)
    print(b)

rando(10)
