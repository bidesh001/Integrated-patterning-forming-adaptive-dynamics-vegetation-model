"""

Code to solve the BWH model

A pseudo-spectral method. Time integration was performed using a fourth 
order Runge Kutta scheme.

@authors: Bidesh K. Bera
"""
import numpy as np
#%% Parameters values
lam=0.032 # growth rate (\lambda_0)
gam=20.0 # water uptake rate (\Gamma)
ns=4.0 # evaporation rate in bare soil (L_0)
rr=10.0  # evaporation reduction due to shading (R)
ff=0.01 # infiltration contrast (f)
qq=0.06 # reference biomass (Q)
aa=40.0 # maximum value of infiltration rate (A)
p=150.00 # precipitation (P)
chi=1.0 # tradeoff parameter (\chi)
kmin=0.1  # minimal capacity to capture light (K_min)
kmax=0.6 # maximal capacity to capture light (K_max)
mmin=0.5 # minimal mortality rate (M_min)
mmax=0.9 # maximal mortality rate (M_max)
ymin=0.5 # minimal contribution to infiltration rate (Y_min)
ymax=1.5 # maximal contribution to infiltration rate (Y_max)
kchi=kmax+(chi)*(kmin-kmax) # tradeoff relation through K
mchi=mmax+(chi)*(mmin-mmax) # tradeoff relation through M
ychi=ymax+(chi)*(ymin-ymax) # tradeoff relation through Y

deltab=1.0  # biomass dispersal rate (D_B)
deltaw=100.0 # soil-water diffusion coefficient (D_W)
deltah=10000.0 # overland-water diffusion coefficient (D_H) 
 
dt=0.01 # step size
NN=128*8 # number of spatial grid points
domainsize=800 # length of spatial domain
tmax=4000 # end time
steps=int(tmax/dt) # time step

L=domainsize/2
x=(2*L/NN)*np.arange(-NN/2,NN/2,1)
real_space=np.linspace(0,domainsize,NN)

f = lambda b,w,h: lam*(1.0-(b/(b+kchi)))*w*b-mchi*b
g = lambda b,w,h: aa*((ychi*b+ff*qq)/(ychi*b+qq))*h-((ns*w)/(1.0+rr*b))-gam*w*b
k = lambda b,w,h: p-aa*((ychi*b+ff*qq)/(ychi*b+qq))*h
#%% Initialise
b = np.ones([NN])*0.12 + np.random.uniform(0,0.001,(NN))
w = np.ones([NN])*0.18 + np.random.uniform(0,0.001,(NN))
h = np.ones([NN])*0.24 + np.random.uniform(0,0.001,(NN))
#%% Construct spatial frequency domain for spectral method
bhat=np.fft.fft(b)
what=np.fft.fft(w)
hhat=np.fft.fft(h)
kx=(np.pi/L)*np.concatenate((np.arange(0,NN/2 +1),np.arange(-NN/2 +1,0)))
ksq=kx**2
#%% Runge–Kutta Method
print_every = 0.1*steps
Eb=np.exp(-deltab*dt*ksq/2)
Eb2=Eb**2
Ew=np.exp(-deltaw*dt*ksq/2)
Ew2 = Ew**2
Eh=np.exp(-deltah*dt*ksq/2)
Eh2 = Eh**2

for n in range(1,steps):        
        
        dbi, dwi, dhi = f(b,w,h), g(b,w,h), k(b,w,h)
        k1b, k1w, k1h = dt*np.fft.fft(dbi), dt*np.fft.fft(dwi), dt*np.fft.fft(dhi)
        b2=np.real(np.fft.ifft(Eb*(bhat+k1b/2)))
        w2=np.real(np.fft.ifft(Ew*(what+k1w/2)))
        h2=np.real(np.fft.ifft(Eh*(hhat+k1h/2)))
     
        dbi,dwi,dhi = f(b2,w2,h2), g(b2,w2,h2), k(b2,w2,h2)
        k2b, k2w, k2h = dt*np.fft.fft(dbi), dt*np.fft.fft(dwi), dt*np.fft.fft(dhi)
        b3=np.real(np.fft.ifft(Eb*bhat+k2b/2)) 
        w3=np.real(np.fft.ifft(Ew*what+k2w/2))
        h3=np.real(np.fft.ifft(Eh*hhat+k2h/2))
     
        dbi,dwi,dhi = f(b3,w3,h3), g(b3,w3,h3), k(b3,w3,h3)
        k3b, k3w, k3h = dt*np.fft.fft(dbi), dt*np.fft.fft(dwi), dt*np.fft.fft(dhi)
        b4=np.real(np.fft.ifft(Eb2*bhat+Eb*k3b))
        w4=np.real(np.fft.ifft(Ew2*what+Ew*k3w))
        h4=np.real(np.fft.ifft(Eh2*hhat+Eh*k3h))
     
        dbi,dwi,dhi = f(b4,w4,h4), g(b4,w4,h4), k(b4,w4,h4)
        k4b, k4w, k4h = dt*np.fft.fft(dbi), dt*np.fft.fft(dwi), dt*np.fft.fft(dhi)
        bhat=Eb2*bhat+(Eb2*k1b+2*Eb*(k2b+k3b)+k4b)/6
        what=Ew2*what+(Ew2*k1w+2*Ew*(k2w+k3w)+k4w)/6
        hhat=Eh2*hhat+(Eh2*k1h+2*Eh*(k2h+k3h)+k4h)/6

        b=np.real(np.fft.ifft(bhat)); w=np.real(np.fft.ifft(what)); h=np.real(np.fft.ifft(hhat))       
          
        if (n % print_every) == 0:
            print(int(100*n/steps))
#np.savez('Singlespecies_P_150',biomass=b) # To save data
