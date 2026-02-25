import numpy as np

def read():
    lst = []
    with open('Pin_smooth.txt') as f:
        lines = f.readlines()
        for i in lines:
            lst.append(float(i.strip(' ').strip('\n').strip('\r')))
    pin = np.array(lst)
    pout = np.ones(len(pin))
    pt = np.ones(len(pin))
    with open('Pout_smooth.txt') as f:
        lines = f.readlines()
        for i in range(len(pin)):
            pout[i] = float(lines[i])
            
    with open('PT_smooth.txt') as f:
        lines = f.readlines()
        for i in range(len(pin)):
            pt[i] = float(lines[i])
    
    return pin,pout,pt
read()