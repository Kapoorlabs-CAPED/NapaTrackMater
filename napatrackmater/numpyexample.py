import numpy as np 

def test_var():
    
    shape = (100000)
    x = (-1 * np.random.random_sample(shape))
    print(x, np.mean(x), np.var(abs(x)))
    
    
    
if __name__=='__main__':
    test_var()    