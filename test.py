import asyncio
import time

def gen():
    i = 1
    for j in range(1, 10):
        print("kek")
        yield j*j
    return 0

a = gen()
print(list(a))

