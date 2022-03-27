import random
import time


def init():
    in1 = 10
    print('bello')

def api():
    print('bleh')
    x = random.randint(0,15)
    y = random.randint(0,15)
    z = random.randint(0,15)
    return x,y,z
def set_motor(spedd,dir,day):
    #todo vymyslet lépe
    in1 = 10
    print(f"{spedd},{dir},{day}")

def zastav


while True:
    try:
        x,y,z = api()
        set_motor(x,y,z)
        time.sleep(1)
    except:
        print("neco se poto")



