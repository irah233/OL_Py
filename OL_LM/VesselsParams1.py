import numpy as np
import scipy.interpolate as sci
from scipy.interpolate import CubicSpline as cub

um2m = 1e-6;  # micron to meter
um2mm = 1e-3; # micron to millimeter
mmHg2Pa = 133.32239; # pressure in mmHg to Pascals
mmHg2MPa = 0.00013332239;# pressure in mmHg to MegaPascals
Pa2MPa = 1e-6; # pressure from Pascal to MegaPascal
dynpcm22MPa =  1.0e-7; # pressure in dynes/cm2 to MegaPascals
dynpcm22Pa = 0.1;
mm3ps2mLpmin = 6e-2; # flow in millimeter3/second to milliliter/minute


def spline(x,y,r):
    spline_ap = cub(x,y)
    y_eval = spline_ap(r)
    return y_eval
    
def pchip(x,y,r):
    pchip_interpolator = sci.PchipInterpolator(x, y)
    y_eval = pchip_interpolator(r)
    return y_eval

def network(n):
    matrix = np.loadtxt("network_matrix_test_400vessel.txt")
    matrix = np.array(matrix)
    long_dist = dict()
    # Initialize vessel parameters
    '''long_dist['ap'] = [0]*n;
    long_dist['bp']= [0]*n;
    long_dist['php'] = [0]*n;
    long_dist['cp']= [0]*n;
    long_dist['rhoa']= [0]*n;
    long_dist['pha']= [0]*n;
    long_dist['ca']= [0]*n;
    long_dist['ma']= [0]*n;
    long_dist['fmax']= [0]*n;
    long_dist['ktau']= [0]*n;
    long_dist['ka']= [0]*n;
    long_dist['a']= [0]*n;'''
    
    data_p = np.array([[3.1,	3.25,	1.0,	0,	19.28],
    [4.7,	4.55,	1.5,	0,	17.24],
    [33.12,	35.02,	10.9,	0.612,	20.11],
    [49.37,	51.38,	16.41,	1.88,	14.23],
    [81.04,	85.53,	32.17,	1.64,	21.24],
    [125.97,	133.52,	51.6+5,	0.96,	23.54],
    [730.5,	829.6,	401.52,	30.39,	3.35]])
    
    rp = data_p[:,0];
    ap = data_p[:,1];
    bp = data_p[:,2];
    php = data_p[:,3];
    cp = data_p[:,4];

        
    # See excel workbook long_distrib_active_params.xls: 
    # R @80mmHg, Rhoa, Phia, ca, fmax, ktau, ka, a
    data = np.array([[3.1,     0,           0,          10,       0,        200,      1.0e-10,     0.3333],
        [4.7,     2.75 ,      20,        10.0,      0.28,       150,     1.98e-09,    0.3333],    
        [33.1,	 26.47,    	69.39,    57.21-10,	    0.43,	    156,     1.98E-09,	0.3333],
        [49.4,	 50.96,  	125.4,	   87.33-10,    0.83,	  199.5,	 4.11E-08,	0.5714],
        [81 ,     77.86,	148.87,	    135.4,  	1,         67.5,     3.55E-07,    0.5714],
        [126 ,    74.14,	77.26,	    51.31,     0.62,        117,     2.15E-07,	0.5714],
        [730.5,	   0,           0 ,      20,       0.0,         200,     1.00E-20,	0.01]])
    ra = data[:,0];
    rhoa = data[:,1];
    pha = data[:,2];
    ca = data[:,3];
    fmax = data[:,4];
    ktau = data[:,5];
    ka = data[:,6];
    a = data[:,7];
    
    r = matrix[:,2]/2;
    r = np.array(r)
    
    Sap = spline(rp,ap,r)
    Sbp = spline(rp,bp,r);
    Sphp = spline(rp,php,r);
    Scp = spline(rp,cp,r);
    Srhoa = pchip(ra,rhoa,r);
    Spha = spline(ra,pha,r);
    Sca = spline(ra,ca,r);
    Sfmax = spline(ra,fmax,r);
    Sktau = spline(ra,ktau,r);
    Ska = spline(ra,ka,r);
    Sa = spline(ra,a,r);
    
    long_dist['ap'] = Sap*um2mm;
    long_dist['bp'] = Sbp*um2mm;
    long_dist['php'] = Sphp*mmHg2MPa;
    long_dist['cp'] = Scp*mmHg2MPa;
    long_dist['rhoa'] = Srhoa*um2mm;
    long_dist['pha'] = Spha*mmHg2MPa;
    long_dist['ca'] = Sca*mmHg2MPa;
    long_dist['ma'] = [2]*n;
    long_dist['fmax'] = Sfmax;
    long_dist['ktau'] = Sktau*dynpcm22MPa;
    long_dist['ka'] = Ska;
    long_dist['a'] = Sa;
    
    for i in range(n):
        if matrix[i][3]<1:
            long_dist['rhoa'][i] = 0;
            long_dist['pha'][i] = 0;
            long_dist['ca'][i] = 20*mmHg2MPa;
            long_dist['fmax'][i] = 0;
            long_dist['ktau'][i] = 1000*dynpcm22MPa;
            long_dist['ka'][i] = 1e-18;
            long_dist['a'][i] = 1e-10;
        #Precautionary measure to bound the parameters by making them positive
        if(long_dist['bp'][i]<0):
            long_dist['bp'][i]=0;
    
        if(long_dist['cp'][i]<0):
            long_dist['cp'][i]=1e-4;
        
        
        if(long_dist['php'][i]<0):
            long_dist['php'][i]=0;
        
        
        if(long_dist['rhoa'][i]<0):
            long_dist['rhoa'][i]=1e-4;
        
        
        if(long_dist['ca'][i]<0):
            long_dist['ca'][i]=1e-4;
        
        
        if(long_dist['fmax'][i]<0):
            long_dist['fmax'][i]=0;
        
        
        if(long_dist['ka'][i]<0):
            long_dist['ka'][i]=1e-10;
        
        
        if(long_dist['a'][i]<=0):
            long_dist['a'][i]=0.01;
            
    return long_dist,matrix
    