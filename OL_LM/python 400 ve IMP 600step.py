from scipy.integrate import odeint,solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import math as ma
import scipy.io as sio
import VesselsParams1 as vp 
from timeit import default_timer
import initial_variable as iv 
from scipy import interpolate as ie
import sympy as sym
from scipy.io import loadmat

font1 = {'weight' : 'normal',
'fontname' : 'Times New Roman',
'size' : 24,}

def  node_(Pm,da1,da2,da3,n,G,Pin,Pout):
    node = np.zeros((401,len(Pin)))
    node[0,:]=(Pin)
    for k in range(1,n+1):
            Git = G[k-1];
            Pmit = Pm[k-1];
            G1 = 0;
            G2 = 0;
            G3 = 0;
            Pm1 = 0 ;
            Pm2 = 0;
            Pm3 = 0;
            if int(da1[k-1])!=-1:
                G1 = G[int(da1[k-1])];
                Pm1 = Pm[int(da1[k-1])];
                if da2[k-1]!=-1:
                    G2 = G[int(da2[k-1])];
                    Pm2 = Pm[int(da2[k-1])];
                    if da3[k-1]!=-1 :
                        G3 = G[int(da3[k-1])];
                        Pm3 = Pm[int(da3[k-1])];
                node[k,:]=((G1*Pm1+G2*Pm2+G3*Pm3+Git*Pmit)/(G1+G2+G3+Git))
                
            if int(da1[k-1])==-1:
                node[k,:]=(Pout)
    return node

def fQin(Pin, Pm, Pnode,G,n,pn):
    #Qi = [0]*n
    Qi = np.zeros((n,len(Pin)))
    Qi = 2*(Pnode[pn,:]-Pm)*G
    return Qi

def fQout(Pout, Pm, Pnode,G,n,dn,da1):
    #Qo = [0]*n
    Qo = np.zeros((n,len(Pout)))
    Qo = 2*(Pm - Pnode[dn,:])*G;
    return Qo

def Bvec_(Pin,Pout,dPTdt,f,da1,G,C,num,fm):
    Bvec = np.ones(num)*dPTdt[f]
    Bvec[0] += 2*Pin[f]*G[0][f]/C[0][f]
    Bvec[fm.Gt_loc[:]] += 2*Pout[f]*G[fm.Gt_loc,f]/C[fm.Gt_loc,f]
    return Bvec

def Amatrix_(G,C,Pin,Pout,j,mo,si,si2,da1,da2,da3,num,fm):
    
    dPt[:,j]=Pm[:,j]-PT[j]
    R0[:,j]=bp[:] + (ap[:]-bp[:])/pi*(pi/2+ np.arctan((dPt[:,j]-php[:])/cp[:]))
    drddeltap_[:,j]=(ap[:]-bp[:])/pi/(1+((dPt[:,j]-php[:])/cp[:])**2)/cp[:]
    R[:,j]=8*mu*L[:]/pi/(R0[:,j])**4
    C[:,j]=2*pi*R0[:,j]*L[:]*drddeltap_[:,j]
    G[:,j]= 1/R[:,j]
        
    
    Amatrix = np.zeros((num,num))
    #print(Bvec)
    
    Amatrix -= np.eye(num)*2
    Git = G[fm.Gnt_loc,j];
    Cit = C[fm.Gnt_loc,j];
    Gm = G[fm.mo,j];
    Gs1 = G[fm.si,j];
    Gs2 = np.zeros(len(fm.Gnt_loc))
    Gs2[fm.si2] = G[fm.si2index,j];
    Gd1 = G[fm.da1,j];
    Gd2 = G[fm.da2,j];
    Gd3 = np.zeros(len(fm.Gnt_loc));
    Gd3[fm.da3] = G[fm.da3index,j];
    Gwm = 1/(Git+Gm+Gs1+Gs2);
    Gwd = 1/(Git+Gd1+Gd2+Gd3);
    #########################internal
    Amatrix[fm.Gnt_loc,fm.Gnt_loc] += Git*Gwm + Git*Gwd;
    Amatrix[fm.Gnt_loc,fm.da1] += (Gd1*Gwd);
    Amatrix[fm.Gnt_loc,fm.da2] += (Gd2*Gwd);
    Amatrix[fm.Gnt_loc[fm.da3],fm.da3index]+= (Gd3[fm.da3]*Gwd[fm.da3]);
    Amatrix[fm.Gnt_loc,fm.mo]  += (Gm*Gwm);
    Amatrix[fm.Gnt_loc,fm.si]  += (Gs1*Gwm);
    Amatrix[fm.Gnt_loc[fm.si2],fm.si2index] += (Gs2[fm.si2]*Gwm[fm.si2]);
    Amatrix[fm.Gnt_loc,:] *= (np.tile((Git/Cit),(num,1))).T;
    ############################################################terminal
    Git = G[fm.Gt_loc,j];
    Cit = C[fm.Gt_loc,j];
    Gs1 = G[fm.si_t,j];
    Gs2 = np.zeros(len(fm.Gt_loc));
    Gs2[fm.si2_t] = (G[fm.si2tindex,j]);
    Gm = G[fm.mo_t,j];
    Gwm = 1/(Git+Gm+Gs1+Gs2);
    Amatrix[fm.Gt_loc,fm.Gt_loc] += (Git*Gwm);
    Amatrix[fm.Gt_loc,fm.mo_t] += (Gm*Gwm);
    Amatrix[fm.Gt_loc,fm.si_t] += (Gs1*Gwm);
    Amatrix[fm.Gt_loc[fm.si2_t],fm.si2tindex] += (Gs2[fm.si2_t]*Gwm[fm.si2_t]);
    Amatrix[fm.Gt_loc,:]*= (np.tile((Git/Cit),(num,1))).T;
    ############################################################source
    Amatrix[0,0] += G[0,j]/(G[0,j]+G[fm.Gnt_loc[0],j]+G[fm.Gnt_loc[1],j]);
    Amatrix[0,fm.Gnt_loc[0]] += G[fm.Gnt_loc[0],j]/(G[0,j]+G[fm.Gnt_loc[0],j]+G[fm.Gnt_loc[1],j]);
    Amatrix[0,fm.Gnt_loc[1]] += G[fm.Gnt_loc[1],j]/(G[0,j]+G[fm.Gnt_loc[0],j]+G[fm.Gnt_loc[1],j]);
    Amatrix[1,:] *= (G[1,j]/C[1,j]);
    Amatrix *= 2
    return Amatrix

def odesystem(t,y,Pm,PT,ap,bp,cp,php,j,num,fm,mo,si,si2,da1,da2,da3,Pin,Pout,dPTdt):

    Pm = y
    dPt=Pm-PT*np.ones(len(Pm[:]))
    R0=bp[:] + (ap[:]-bp[:])/pi*(pi/2+ np.arctan((dPt-php[:])/cp[:]))
    drddeltap_=(ap[:]-bp[:])/pi/(1+((dPt-php[:])/cp[:])**2)/cp[:]
    R=8*mu*L[:]/pi/(R0)**4
    C=2*pi*R0*L[:]*drddeltap_
    G= 1/R
    
    flow_vector = np.ones(num)*dPTdt
    flow_vector[0] += 2*Pin[j]*G[0]/C[0]
    flow_vector[fm.Gt_loc[:]] += 2*Pout[j]*G[fm.Gt_loc]/C[fm.Gt_loc]
    n = len(Pm)
    flow_matrix = np.zeros((n));
    Gm = np.zeros((n));
    Gs = np.zeros((n));
    nnt = len(fm.Gnt_loc);
    nt = len(fm.Gt_loc);
    Gs2 = np.zeros((nnt));
    Gsi2_t = np.zeros((nt));
    Gd1 = np.zeros((n));
    Gd2 = np.zeros((n));
    Gd3 = np.zeros((nnt));
    Pcm = np.zeros((n));
    Pcs = np.zeros((n));
    Pcs2 = np.zeros((nnt));
    Pcs2_t = np.zeros((nt));
    Pcd1 = np.zeros((n));
    Pcd2 = np.zeros((n));
    Pcd3 = np.zeros((nnt));
    Gm = G[fm.mo];
    Gs = G[fm.si];
    Gs2[fm.si2] = G[fm.si2index];
    Gd1 = G[fm.da1];
    Gd2 = G[fm.da2];
    Gd3[fm.da3] = G[fm.da3index];
    Gnt = G[fm.Gnt_loc];
    Cnt = C[fm.Gnt_loc];
    Gm_t  = G[fm.mo_t];
    Gs_t = G[fm.si_t];
    Gsi2_t[fm.si2_t] = G[fm.si2tindex];
    G_t = G[fm.Gt_loc];
    C_t = C[fm.Gt_loc];
    Pcm = Pm[fm.mo];
    Pcs = Pm[fm.si];
    Pcs2[fm.si2] = Pm[fm.si2index];
    Pcd1 = Pm[fm.da1];
    Pcd2 = Pm[fm.da2];
    Pcd3[fm.da3] = Pm[fm.da3index];
    Pc_nt = Pm[fm.Gnt_loc];
    Pcm_t = Pm[fm.mo_t];
    Pcs_t = Pm[fm.si_t];
    Pcs2_t[fm.si2_t]=Pm[fm.si2tindex];
    Pc_t = Pm[fm.Gt_loc];
    one_by_den1 = 1/(Cnt*(Gnt+Gm+Gs+Gs2));
    one_by_den2 = 1/(Cnt*(Gnt + Gd1+ Gd2 + Gd3));
    flow_matrix[fm.Gnt_loc] = (2*Gm*Gnt*Pcm*one_by_den1) + (2*Gs*Gnt*Pcs*one_by_den1)+  \
    (2*Gs2*Gnt*Pcs2*one_by_den1)+ \
    (-4*Gnt*Pc_nt/Cnt + 2*Gnt*Gnt*Pc_nt*one_by_den1+ \
    2*Gnt*Gnt*Pc_nt*one_by_den2)+ \
    (2*Gd1*Gnt*Pcd1*one_by_den2)+ \
    (2*Gd2*Gnt*Pcd2*one_by_den2)+ \
    (2*Gd3*Gnt*Pcd3*one_by_den2);
    one_by_den3 = 1/(C_t*(G_t+Gm_t+Gs_t+Gsi2_t));
    flow_matrix[fm.Gt_loc] = (2*Gm_t*G_t*Pcm_t*one_by_den3) + \
    (2*Gs_t*G_t*Pcs_t*one_by_den3)+\
    (2*Gsi2_t*G_t*Pcs2_t*one_by_den3)+\
    (-4*G_t*Pc_t/C_t + 2*G_t*G_t*Pc_t*one_by_den3);
    flow_matrix[0] =  (-4*G[0]*Pm[0]/C[0] + \
    2*G[0]*G[0]*Pm[0]/(C[0]*(G[0] + Gd1[0]+ Gd2[0])))+ \
    (2*Gd1[0]*G[0]*Pcd1[0]/(C[0]*(G[0] + Gd1[0]+Gd2[0])))+ \
    (2*Gd2[0]*G[0]*Pcd2[0]/(C[0]*(G[0]+Gd1[0]+Gd2[0])));

    
    dy_dt = flow_matrix + flow_vector
    #print(duration)
    #print(sym.Matrix(Amatrix))
    return dy_dt

array = np.arange(0.2,1.4,0.2)
print(array)

pin_scale_lst = array
for pin_scale in pin_scale_lst:


            mmHg2MPa = 0.0001333223684;
            Pin = []  
            Pout = []  
            PT = [] 
            '''
            mat_data = sio.loadmat('Pa_RA.mat')
            Pin = mat_data['Pa'][0]*mmHg2MPa
            #Pin = [Pin[0]]*len(Pin)
            mat_data = sio.loadmat('Pv_RA.mat')
            Pout = mat_data['Pv'][0]*mmHg2MPa
            #Pout = [Pout[0]]*len(Pin)
            mat_data = sio.loadmat('IMP_LAD_RA.mat')
            PT = mat_data['IMP_LAD'][0][:600]*mmHg2MPa
            #PT = [0]*len(Pin)#mat_data['IMP_LAD'][0]*1e-6
            #mat1 = sio.loadmat('mat_400v_IMP.mat')
            #Pm1 = mat1['Pm']
            with open("Pin_smooth.txt") as f:
                for line in f:
                    Pin.append(float(line[:5])*(10**int(line[-2])))
            
            with open("Pout_smooth.txt") as f:
                for line in f:
                    Pout.append(float(line[:5])*(10**int(line[-2])))
            with open("Pt_smooth.txt") as f:
                for line in f:
                    PT.append(float(line[:5])*(10**int(line[-2])))
            '''
            Pressure_data=loadmat('Part_new.mat')
            Pin = Pressure_data['Part']
            Pin_data=np.asfarray(Pin,float)*1e-6
            Pin=Pin_data[0,:]#
            Pin = Pin[:len(Pin)*1//3]
            #Pin = max(Pin)*np.ones(len(Pin))
            
            Pressure_data=loadmat('Pven_new.mat')
            Pout = Pressure_data['Pven']
            Pout_data=np.asfarray(Pout,float)*1e-6
            Pout=Pout_data[0,:]
            Pout = Pout[:len(Pout)*1//3]
            #Pout = Pout[0]*np.ones(len(Pout))
            
            Pressure_data=loadmat('IMPLAD_new.mat')
            PT = Pressure_data['IMPLAD']
            PT_data=np.asfarray(PT,float)*1e-6
            Pt=PT_data[0,:]
            Pt = Pt[:len(Pt)*1//3]#/2
            #Pt=np.zeros(len(Pt))
                
            start1 = default_timer()
            
            
            Pin = np.array(Pin)*pin_scale
            Pout = np.array(Pout)
            PT = np.array(Pt)
            num = 400
            matrix, network = vp.network(400)
            fm = iv.initial_variable(network)
            ap = np.array(matrix['ap'][:num])
            bp = np.array(matrix['bp'][:num])
            cp = np.array(matrix['cp'][:num])
            php = np.array(matrix['php'][:num])
            
            L = network[:num,1]*1e-3
            
            
            
            pn = list(map(int, np.array(network[:,4])-1));         # proximal node
            dn = list(map(int, np.array(network[:,5])-1));         # distal node
            mo = list(map(int, np.array(network[:,6])));         # mother index
            si = list(map(int, np.array(network[:,7])));         # sister1
            si2 = list(map(int, np.array(network[:,8])));        # sister2
            da1 = list(map(int, np.array(network[:,11])-1));       # daughter1
            da2 = list(map(int, np.array(network[:,12])-1));       # daughter2
            da3 = list(map(int, np.array(network[:,13])-1));       # daughter3 if any
            
            
            n=400
            mu = 2.7e-9
            num_cyc=2
            pi = ma.pi
            #dt = 1e-5; #0.001;       # unit of time: s
            BCL = 60.0/100.0; # Basic Cycle Length
            t = np.linspace(0, 1*BCL, len(Pout));#(len(Pin)-1)*dt
            t2 = np.linspace(0, 1*BCL, len(PT));
            xx = np.linspace(0, 1*BCL, 1200)
            xx1 = np.linspace(0, 1*BCL, 1201)
            Pin = np.interp(xx ,t , Pin)
            Pout = np.interp(xx ,t , Pout)
            PT = np.interp(xx ,t2 , PT)
            
            
            
            PT1 = np.interp(xx1 ,np.linspace(0, num_cyc*BCL, len(PT)) , PT)
            t1 = np.interp(xx1 ,t , t)
            t = np.interp(xx ,t , t)
            
            dx = np.diff(t1)
            dy = np.diff(PT1)
            dPTdt = dy/dx
            
            R0 = np.zeros((400,len(Pin)))
            Qc = np.zeros((400,len(Pin)))
            eQ = np.zeros((400,len(Pin)))
            GG = np.zeros((400,len(Pin)))
            CC = np.zeros((400,len(Pin)))
            #Pnode1=[[float(0)]*len(Pin)]*3
            Pnode = np.zeros((401,len(Pin)))
            N=len(Pin)
            dPt = np.zeros((400,len(Pin)))
            Qin = np.zeros((400,len(Pin)))
            Qout = np.zeros((400,len(Pin)))
            Pm = 0.0027*np.ones((400,len(Pin)))
            R0 = np.zeros((400,len(Pin)))
            drddeltap_= np.zeros((400,len(Pin)))
            G = np.zeros((400,len(Pin)))
            C = np.zeros((400,len(Pin)))
            R = np.zeros((400,len(Pin)))
            #Pm = np.array(Pm1*mmHg2MPa).T
            t = np.linspace(0, num_cyc*BCL, len(Pin)+1)
            
            for j in range(len(Pin)):
                if j == 0:
                    for mm in range(10):
                        y0 = []
                        
                        '''dPt[:,j]=Pm[:,j]-PT[j]
                        R0[:,j]=bp[:] + (ap[:]-bp[:])/pi*(pi/2+ np.arctan((dPt[:,j]-php[:])/cp[:]))
                        drddeltap_[:,j]=(ap[:]-bp[:])/pi/(1+((dPt[:,j]-php[:])/cp[:])**2)/cp[:]
                        R[:,j]=8*mu*L[:]/pi/(R0[:,j])**4
                        C[:,j]=2*pi*R0[:,j]*L[:]*drddeltap_[:,j]
                        G[:,j]= 1/R[:,j]'''
                        y0=np.array(Pm[:,j])#+np.zeros((1,400))
                        if j>=1:
                            y0=np.array(Pm[:,j-1])
                        print(j+1)       
                        
                        #print(t)
                        start = default_timer()
                        
                        #t_span = (0.0, 0.001)
                        t_span = (t[j],t[j+1])
                        if j>=1:
                            t_span = (t[j-1],t[j])
                        #if j<1:
                        sol = solve_ivp(odesystem, t_span, y0, method='BDF',rtol=1e-6, atol=1e-7,args = (Pm[:,j],PT[j],ap,bp,cp,php,j,num,fm,mo,si,si2,da1,da2,da3,Pin,Pout,dPTdt[j],))
                        #if j>=1:
                        #    sol = solve_ivp(odesystem, t_span, y0, method='BDF',rtol=1e-6, atol=1e-3,args = (Pm[:,j-1],PT[j],ap,bp,cp,php,j,num,fm,mo,si,si2,da1,da2,da3,Pin,Pout,dPTdt[j],))
                        #print(G[:,j])'''
                        #sol = odeint(odesystem, y0, t, args = (G,C,Pin,Pout,j,mo,si,si2,da1,da2,da3,num,))#mxstep=200
                        duration = default_timer() - start
                        print(duration)
                        #print(sol[-1,:])
                            #if (d1-d2)/d1 < 0.005:
                            #    break
                        Pm[:,j] = sol['y'][:,-1];
                else:
                    y0 = []
                    y0=np.array(Pm[:,j-1])
                    print(j+1)       
                    
                    #print(t)
                    start = default_timer()
                    
                    #t_span = (0.0, 0.001)
                    t_span = (t[j],t[j+1])
                    if j>=1:
                        t_span = (t[j-1],t[j])
                    #if j<1:
                    sol = solve_ivp(odesystem, t_span, y0, method='BDF', atol=1e-3,args = (Pm[:,j-1],PT[j],ap,bp,cp,php,j,num,fm,mo,si,si2,da1,da2,da3,Pin,Pout,dPTdt[j],))
                    duration = default_timer() - start
                    print(duration)
                    Pm[:,j] = sol['y'][:,-1];
                    #print(Pm[:,j][0])
                dPt[:,j]=Pm[:,j]-PT[j]
                R0[:,j]=bp[:] + (ap[:]-bp[:])/pi*(pi/2+ np.arctan((dPt[:,j]-php[:])/cp[:]))
                drddeltap_[:,j]=(ap[:]-bp[:])/pi/(1+((dPt[:,j]-php[:])/cp[:])**2)/cp[:]
                R[:,j]=8*mu*L[:]/pi/(R0[:,j])**4
                C[:,j]=2*pi*R0[:,j]*L[:]*drddeltap_[:,j]
                G[:,j]= 1/R[:,j]
                    #print(Pm[0][j])
                    
                    
            Pnode=(node_(Pm,da1,da2,da3,n,G,Pin,Pout))
            
            
            Pm_1 = Pm[0]
            Pm_2 = Pm[1]
            Pm_3 = Pm[2]
            t = np.linspace(0, 1*BCL, len(Pout))
            
            r_1 = R0[0]
            r_2 = R0[1]
            r_3 = R0[2]
            
            
            Qi = fQin(Pin, Pm, Pnode,G,n,pn)
            Qo = fQout(Pout, Pm, Pnode,G,n,dn,da1)
            
            plt.plot(t,Pm_1/mmHg2MPa, label='Pm_1')
            plt.plot(t,Pm_2/mmHg2MPa, label='Pm_2')
            plt.plot(t,Pm_3/mmHg2MPa, label='Pm_3')
            plt.legend(loc='upper right',  frameon=False, ncol=2, fontsize=24)
            plt.yticks(fontsize=24, rotation=0, fontname="Times New Roman")
            plt.xticks(fontsize=24, rotation=0, fontname="Times New Roman")
            plt.xlabel("Time (s)",font1)
            plt.ylabel("pressure (mmHg)",font1)
            #plt.xlim(2.4, 3)
            plt.tight_layout()
            plt.savefig('400v pressure IMP.png')
            plt.close('all')
            
            Qin1 = ((Qi))[0]
            Qin2 = ((Qi))[1]
            Qin3 = ((Qi))[2]
            
            plt.plot(t,Qin1, label='Qin1')
            plt.plot(t,Qin2, label='Qin2')
            plt.plot(t,Qin3, label='Qin3')
            plt.legend(loc='upper right',  frameon=False, ncol=2, fontsize=24)
            plt.yticks(fontsize=24, rotation=0, fontname="Times New Roman")
            plt.xticks(fontsize=24, rotation=0, fontname="Times New Roman")
            plt.xlabel("Time (s)",font1)
            plt.ylabel("Flow Rate $(mm^3/sec)$",font1)
            #plt.xlim(2.4, 3)
            plt.tight_layout()
            plt.savefig('400v flow in IMP.png')
            plt.close('all')
            
            Qout1 = ((Qo))[0]
            Qout2 = ((Qo))[1]
            Qout3 = ((Qo))[2]
            
            plt.plot(t,Qout1, label='Qout1')
            plt.plot(t,Qout2, label='Qout2')
            plt.plot(t,Qout3, label='Qout3')
            plt.legend(loc='upper right',  frameon=False, ncol=2, fontsize=24)
            plt.yticks(fontsize=24, rotation=0, fontname="Times New Roman")
            plt.xticks(fontsize=24, rotation=0, fontname="Times New Roman")
            plt.xlabel("Time (s)",font1)
            plt.ylabel("Flow Rate $(mm^3/sec)$",font1)
            #plt.xlim(2.4, 3)
            plt.tight_layout()
            plt.savefig('400v flow out IMP.png')
            plt.close('all')
            
            plt.plot(t,r_1, label='r_1')
            plt.plot(t,r_2, label='r_2')
            plt.plot(t,r_3, label='r_3')
            plt.legend(loc='upper right',  frameon=False, ncol=2, fontsize=24)
            plt.yticks(fontsize=24, rotation=0, fontname="Times New Roman")
            plt.xticks(fontsize=24, rotation=0, fontname="Times New Roman")
            plt.xlabel("Time (s)",font1)
            plt.ylabel("Rad (mm)",font1)
            #plt.xlim(2.4, 3)
            plt.tight_layout()
            plt.savefig('400v rad IMP.png')
            plt.close('all')
            
            duration2 = default_timer() - start1
            print(duration2)
            
            dic={}
            dic["t"]=t
            dic["Pin"]=Pin/mmHg2MPa
            dic["Pm"]=(Pm)/mmHg2MPa
            dic["Pout"]=Pout/mmHg2MPa
            dic["R0"] = np.array(R0)
            dic["Qi"]=Qi
            dic["Qo"]=Qo
            dic["Pn"]=Pnode
            dic["G"]=G
            dic["C"]=C
            scale1 = ("{:.1f}".format(pin_scale))
            print('python_400v_IMP Pin '+str(scale1)+'.mat')
            sio.savemat('python_400v_IMP Pin '+str(scale1)+'.mat', dic)

