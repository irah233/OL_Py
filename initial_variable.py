import numpy as np

class iv:
    def __init__(self):
        self.Gnt_loc = []
        self.Gt_loc = []
        self.mo = []
        self.si = []
        self.si2 = []
        self.si2index = []
        self.da1 = []
        self.da2 = []
        self.da3 = []
        self.da3index = []
        self.mo_t = []
        self.si_t = []
        self.si2_t = []
        self.si2tindex = []
        self.sis_check = []
        

        
def initial_variable(network):
    mo = network[:,6];         # mother index
    si = network[:,7];         # sister1
    si2 = network[:,8];        # sister2
    da1 = network[:,11];       # daughter1
    da2 = network[:,12];       # daughter2
    da3 = network[:,13];       # daughter3 if any
    fm = iv()
    fm.Gnt_loc = np.array([num for num in range(len(mo)) if da1[num]>0 and mo[num]>0]);    # Non terminal vessels
    fm.Gt_loc = np.array([num for num in range(len(mo)) if da1[num]==0]);         # Terminal vessels
    fm.mo = list(map(int, np.array(mo[fm.Gnt_loc])-1));           # non-terminal vessels' mothers' index          #
    fm.si = list(map(int, np.array(si[fm.Gnt_loc])-1));           # non-terminal vessels' sister1' index
    fm.si2 = [num for num in range(len(fm.Gnt_loc)) if si2[fm.Gnt_loc[int(num)]]>0]; # find veseels who have sister2
    fm.si2index = [int(num-1) for num in si2[fm.Gnt_loc] if num>0 ];
    fm.da1 = list(map(int, np.array(da1[fm.Gnt_loc])-1));         # non-terminal vessels' daughter1' index
    fm.da2 = list(map(int, np.array(da2[fm.Gnt_loc])-1));         # non-terminal vessels' daughter2' index
    fm.da3 = [num for num in range(len(fm.Gnt_loc)) if da3[fm.Gnt_loc[int(num)]]>0]; # find vessels who have daughter3
    fm.da3index = [int(num-1) for num in da3[fm.Gnt_loc] if num>0 ]; # find the nonzero daughter3
    fm.mo_t = list(map(int, np.array(mo[fm.Gt_loc])-1));          # terminal vessels' mothers' index
    fm.si_t = list(map(int, np.array(si[fm.Gt_loc])-1));          # terminal vessels' sister1' index
    fm.si2_t = [num for num in range(len(fm.Gt_loc)) if si2[fm.Gt_loc[int(num)]]>0];# terminal vessels' who have sister2
    fm.si2tindex = [int(num-1) for num in si2[fm.Gt_loc] if num>0 ];
    fm.sis_check = list(map(int, np.array(si2[fm.Gt_loc])-1));
    return fm
