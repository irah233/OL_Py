from scipy.integrate import odeint,solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import math
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
    node = np.zeros((401,len(Pm[0])))
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

a_ratio = 0.5
pin_scale_lst = array
rhoa_scale = 1
pha_scale = 0.2
ca_scale = 0.6
fmax_scale = 1
ktau_scale = 20
ka_scale = 1
ma_scale = 1
cp_scale = 1
Femta_input=0.7
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
                matrix, network = vp.network(num)
                fm = iv.initial_variable(network)
                ap = np.array(matrix['ap'][:num])
                bp = np.array(matrix['bp'][:num])
                cp = np.array(matrix['cp'][:num])*cp_scale
                php = np.array(matrix['php'][:num])
                
                rhom = np.array(matrix['rhoa'][:num])*rhoa_scale
                pha = np.array(matrix['pha'][:num])*pha_scale
                ca = np.array(matrix['ca'][:num])*ca_scale
                fmax = np.array(matrix['fmax'][:num])*fmax_scale
                ktau = np.array(matrix['ktau'][:num])*ktau_scale
                ka = np.array(matrix['ka'][:num])*ka_scale
                
                
                L = network[:num,1]*1e-3
                
                indx = loadmat('indx.mat')
                indx = indx['indx']
                cond_leng =  loadmat('cond_leng.mat')
                cond_leng = cond_leng['cond_leng']
                cond_leng = cond_leng/1000;
                
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
                pi = np.pi
                #dt = 1e-5; #0.001;       # unit of time: s
                BCL = 60.0/100.0; # Basic Cycle Length
                t = np.linspace(0, 1*BCL, len(Pout));#(len(Pin)-1)*dt
                t2 = np.linspace(0, 1*BCL, len(PT));
                xx = np.linspace(0, 1*BCL, 1200)
                xx1 = np.linspace(0, 1*BCL, 1201)
                Pin = np.interp(xx ,t , Pin)
                Pout = np.interp(xx ,t , Pout)
                PT = np.interp(xx ,t2 , PT)
                ma = 1
                
                
                PT1 = np.interp(xx1 ,np.linspace(0, num_cyc*BCL, len(PT)) , PT)
                t1 = np.interp(xx1 ,t , t)
                t = np.interp(xx ,t , t)
                
                dx = np.diff(t1)
                dy = np.diff(PT1)
                dPTdt = dy/dx
                
                R0 = np.zeros((num,len(Pin)))
                Qc = np.zeros((num,len(Pin)))
                eQ = np.zeros((num,len(Pin)))
                GG = np.zeros((num,len(Pin)))
                CC = np.zeros((num,len(Pin)))
                #Pnode1=[[float(0)]*len(Pin)]*3
                Pnode = np.zeros((num+1,len(Pin)))
                N=len(Pin)
                dPt = np.zeros((num,len(Pin)))
                Qin = np.zeros((num,len(Pin)))
                Qout = np.zeros((num,len(Pin)))
                Pm = 0.0027*np.ones((num,len(Pin)))
                r_r = np.zeros((len(Pin),num))
                drddeltap_= np.zeros((num,len(Pin)))
                G = np.zeros((num,len(Pin)))
                C = np.zeros((num,len(Pin)))
                R = np.zeros((num,len(Pin)))
                #Pm = np.array(Pm1*mmHg2MPa).T
                
                lst_tau_avg = []#np.zeros((num,len(Pin)))
                lst_A = []#np.zeros((num,len(Pin)))
                lst_dp_avg = []#np.zeros((num,len(Pin)))
                lst_Rm_avg = []#np.zeros((num,len(Pin)))
                tau_value = np.zeros(num)
                
                if Femta_input == 1:
                    Fmeta_data = np.ones(np.shape(ktau))
                else:
                
                    Fmeta_value = Femta_input
                
                    Fmeta_inform = loadmat('indx.mat')
                    Fmeta_indx = Fmeta_inform['indx']
                    Fmeta_indx=np.asfarray(Fmeta_indx,float) 
                    
                    Fmeta_inform1 = loadmat('cond_leng.mat')
                    Fmeta_cond_leng = Fmeta_inform1['cond_leng']
                    Fmeta_cond_leng=np.asfarray(Fmeta_cond_leng,float)/1000    
                
                    index1 = []
                    for Fm in range(0,len(network[:,3])):
                        if network[Fm,3] ==1 and network[Fm,11] ==0:
                            index1.append(Fm)
                    
                    Fn = len(network[:,3])
                    Fmeta_data = np.zeros((Fn,1))
                    FAd1 = Fmeta_value*np.ones((len(index1),1))
                
                    for Fm in range(0,len(index1)):
                        Fmeta_data[index1[Fm]] = FAd1[Fm]
                
                    for Fm in range(0,Fn):
                        term_vess_order1 = Fmeta_indx[Fm,np.nonzero(Fmeta_indx[Fm,:])]
                        term_vess_order1 = term_vess_order1[0]
                        cond_leng_order1 = Fmeta_cond_leng[Fm,np.nonzero(Fmeta_cond_leng[Fm,:])]
                        cond_leng_order1 = cond_leng_order1[0]
                
                        index2 = 0
                        if len(term_vess_order1)!=0:
                            for Fm1 in range(0,len(term_vess_order1)):
                                index2 = np.where(index1[:] == (term_vess_order1[Fm1])-1)
                                Fmeta_data[Fm][0] = Fmeta_data[Fm][0] + FAd1[index2][0]*np.exp(-cond_leng_order1[Fm1]/1)    
                            Fmeta_data[Fm][0] = Fmeta_data[Fm][0]/len(term_vess_order1)
                Fmeta_data = np.concatenate(Fmeta_data, axis=None)
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
                        sol = solve_ivp(odesystem, t_span, y0, method='BDF', atol=1e-9,args = (Pm[:,j-1],PT[j],ap,bp,cp,php,j,num,fm,mo,si,si2,da1,da2,da3,Pin,Pout,dPTdt[j],))
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
                    Pnode=Pnode[:,:j+1]
                    
                    tau_value[0] = (Pin[j] - Pnode[0,j])*R0[0,j]/(2*L[0])
                    tau_value[1:] = abs((Pnode[pn[1:],j]-Pnode[dn[1:],j]))\
                            *R0[1:,j]/(2*L[1:])
                
                    #tau_arr = tau_value
                    lst_tau_avg.append(tau_value)
                
                    tau_avg_arr = abs(np.mean(lst_tau_avg,0))
                    
                    Ftau_value_arr = fmax*tau_avg_arr/(ktau+tau_avg_arr)
                    Ftau_arr = Ftau_value_arr
                
                    for jj in range(len(Ftau_arr)):
                        if Ftau_arr[jj]<0:
                            Ftau_arr[jj]=0
                        elif Ftau_arr[jj]>1:
                            Ftau_arr[jj]=1
                        else:
                            Ftau_arr[jj]=Ftau_arr[jj]
                    
                    A = (1-Ftau_arr) * (1-Fmeta_data) * a_ratio
                    
                    dp_array = dPt[:,j]#.vector().get_local()
                    lst_dp_avg.append(dp_array)
                    dp_avg = np.mean(lst_dp_avg,0)
                
                    Rm_avg = (rhom/np.pi) * ( 0.5*np.pi - np.arctan( ( (dp_avg - pha)/ca)**(2*ma) ) )
                    
                    lst_A.append(A)
                    lst_Rm_avg.append(Rm_avg)
                    r_r[j,:] = R0[:,j] - A * Rm_avg
                    
                    R[:,j]=8*mu*L[:]/pi/(r_r[j,:])**4
                    G[:,j]= 1/R[:,j]
                    
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
                plt.savefig('400v pressure reg.png')
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
                plt.savefig('400v flow in reg.png')
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
                plt.savefig('400v flow out reg.png')
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
                plt.savefig('400v rad reg.png')
                plt.close('all')
                
                lst_tau_avg = np.array(lst_tau_avg)
                plt.plot(t,lst_tau_avg[:,0], label='tau_1')
                plt.plot(t,lst_tau_avg[:,1], label='tau_2')
                plt.plot(t,lst_tau_avg[:,2], label='tau_3')
                plt.legend(loc='upper right',  frameon=False, ncol=2, fontsize=24)
                plt.yticks(fontsize=24, rotation=0, fontname="Times New Roman")
                plt.xticks(fontsize=24, rotation=0, fontname="Times New Roman")
                plt.xlabel("Time (s)",font1)
                plt.ylabel("tau (mm)",font1)
                #plt.xlim(2.4, 3)
                plt.tight_layout()
                plt.savefig('400v tau reg.png')
                plt.close('all')
                
                lst_A = np.array(lst_A)
                plt.plot(t,lst_A[:,0], label='A_1')
                plt.plot(t,lst_A[:,1], label='A_2')
                plt.plot(t,lst_A[:,2], label='A_3')
                plt.legend(loc='upper right',  frameon=False, ncol=2, fontsize=24)
                plt.yticks(fontsize=24, rotation=0, fontname="Times New Roman")
                plt.xticks(fontsize=24, rotation=0, fontname="Times New Roman")
                plt.xlabel("Time (s)",font1)
                plt.ylabel("A (mm)",font1)
                #plt.xlim(2.4, 3)
                plt.tight_layout()
                plt.savefig('400v A reg.png')
                plt.close('all')
                
                lst_dp_avg = np.array(lst_dp_avg)
                plt.plot(t,lst_dp_avg[:,0], label='dp_1')
                plt.plot(t,lst_dp_avg[:,1], label='dp_2')
                plt.plot(t,lst_dp_avg[:,2], label='dp_3')
                plt.legend(loc='upper right',  frameon=False, ncol=2, fontsize=24)
                plt.yticks(fontsize=24, rotation=0, fontname="Times New Roman")
                plt.xticks(fontsize=24, rotation=0, fontname="Times New Roman")
                plt.xlabel("Time (s)",font1)
                plt.ylabel("dp (mm)",font1)
                #plt.xlim(2.4, 3)
                plt.tight_layout()
                plt.savefig('400v dp reg.png')
                plt.close('all')
                
                lst_Rm_avg = np.array(lst_Rm_avg)
                plt.plot(t,lst_Rm_avg[:,0], label='Rm_avg_1')
                plt.plot(t,lst_Rm_avg[:,1], label='Rm_avg_2')
                plt.plot(t,lst_Rm_avg[:,2], label='Rm_avg_3')
                plt.legend(loc='upper right',  frameon=False, ncol=2, fontsize=24)
                plt.yticks(fontsize=24, rotation=0, fontname="Times New Roman")
                plt.xticks(fontsize=24, rotation=0, fontname="Times New Roman")
                plt.xlabel("Time (s)",font1)
                plt.ylabel("Rm_avg (mm)",font1)
                #plt.xlim(2.4, 3)
                plt.tight_layout()
                plt.savefig('400v Rm_avg reg.png')
                plt.close('all')
                
                duration2 = default_timer() - start1
                print(duration2)
                
                dic={}
                dic["t"]=t
                dic["Pin"]=Pin/mmHg2MPa
                dic["Pm"]=(Pm)/mmHg2MPa
                dic["Pout"]=Pout/mmHg2MPa
                dic["rp"] = np.array(R0)
                dic["rr"] = np.array(r_r)
                dic["Qi"]=Qi
                dic["Qo"]=Qo
                dic["Pn"]=Pnode
                dic["G"]=G
                dic["C"]=C
                scale1 = ("{:.1f}".format(pin_scale))
                print('python_400v_reg Pin '+str(scale1)+"_Cp_"+str(cp_scale)+"_ktau_scale"+str(ktau_scale)\
                        +"_rhoa"+str(rhoa_scale)+"_fmax"+str(fmax_scale)+"_Femta"+str(Femta_input)\
                        +"_pha"+str(pha_scale)+"_ca"+str(ca_scale)+"_ma"+str(ma_scale))
                sio.savemat('python_400v_reg Pin '+str(scale1)+"_Cp_"+str(cp_scale)+"_ktau_scale"+str(ktau_scale)\
                        +"_rhoa"+str(rhoa_scale)+"_fmax"+str(fmax_scale)+"_Femta"+str(Femta_input)\
                        +"_pha"+str(pha_scale)+"_ca"+str(ca_scale)+"_ma"+str(ma_scale)+'.mat', dic)

fmax_scale = [1] # value Must consistent with 'Preprocess.py'
pha_scale = [0.2] # value Must consistent with 'Preprocess.py'
ca_scale = [0.5] # value Must consistent with 'Preprocess.py'
ma_scale = [1] # value Must consistent with 'Preprocess.py'
rhoa_value = [0.3] # value Must consistent with 'Preprocess.py'
ktau_scale = [20] # value Must consistent with 'Preprocess.py'
