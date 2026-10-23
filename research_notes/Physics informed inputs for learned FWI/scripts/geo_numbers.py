import numpy as np
fp=15.0
print("Wavelengths at 15 Hz and fmax~2.5fp")
for v in [1500,2000,3000,4500]:
    lam=v/fp; print(v, "lam=%.0f lam/2=%.0f lam/4=%.0f lam/8=%.0f  lam@37.5Hz=%.0f"%(lam,lam/2,lam/4,lam/8,v/37.5))
# Ricker amplitude spectrum relative to peak: A(f)=(f/fp)^2 exp(1-(f/fp)^2)
print("Ricker spectrum rel. to peak (dB)")
for f in [1,2,3,4,5,7.5,10,30,40,45,50]:
    r=(f/fp)**2*np.exp(1-(f/fp)**2); print(f, "%.4f  %.1f dB"%(r,20*np.log10(r)))
# Diving wave turning depth for linear gradient v=v0+g z: z_t = sqrt((x/2)^2+(v0/g)^2)-v0/g
print("Diving turning depth, v0=1500")
for g in [0.5,1,2,4.29]:
    for x in [350,700]:
        a=1500/g; zt=np.sqrt((x/2)**2+a**2)-a
        print("g=%.2f/s x=%d m -> z_t=%.0f m (v at z_t=%.0f)"%(g,x,zt,1500+g*zt))
# check SEP example v0=2000 g=1
a=2000/1.0
for x in [4470,6930]: print("SEP check",x,np.sqrt((x/2)**2+a**2)-a)
# critical angle and crossover for two-layer
print("two-layer refraction: h, v1, v2 -> theta_c, x_crit, x_cross")
for h,v1,v2 in [(100,1500,2000),(200,1500,2500),(200,2000,3000),(300,2000,4000),(150,1500,3000),(100,1500,4500)]:
    tc=np.arcsin(v1/v2); xc=2*h*np.tan(tc); xco=2*h*np.sqrt((v2+v1)/(v2-v1))
    print(h,v1,v2,"theta_c=%.1f deg x_crit=%.0f x_cross=%.0f"%(np.degrees(tc),xc,xco))
# scattering angle at depth z for half-offset h in homogeneous medium; specular reflection opening angle
print("max opening angle at depth z (flat reflector, offset 700 and 350)")
for z in [100,200,300,500,700]:
    for off in [700,350]:
        th=2*np.arctan(off/2/z)
        print(z,off,"open=%.0f deg incidence=%.0f deg cos(th/2)=%.2f"%(np.degrees(th),np.degrees(th/2),np.cos(th/2)))
# Wavenumber coverage: vertical wavelength resolved lambda_z = v/(2 f cos(th/2))
print("resolved vertical wavelength range for reflections at z=500, v=3000, f=3..40Hz, open angle 0..70deg")
v=3000
for f in [3,5,15,40]:
    for th in [0,70]:
        k=2*f*np.cos(np.radians(th)/2)/v; print(f,th,"lambda_res=%.0f m"%(1/k))
# Fresnel zone radius unmigrated sqrt(z lambda/2)
print("Fresnel radius sqrt(z*lam/2) unmigrated, v=2500 lam=167")
for z in [100,300,500,700]:
    print(z, "%.0f"%np.sqrt(z*2500/15/2))
# Fresnel width of diving ray path sqrt(lam L)/2 half width midpoint sqrt(L lam/4)
for L,v in [(350,1500),(700,2000)]:
    lam=v/fp; print("diving Fresnel half-width midpoint L=%d v=%d: %.0f m"%(L,v,np.sqrt(L*lam/4)))
# cycle skip: traveltime error < T/2 = 1/(2*15)=33 ms. Relative velocity error threshold for N wavelengths: dv/v < 1/(2N)
print("T/2 at 15 Hz = %.1f ms; at 5 Hz = %.0f ms; at 3 Hz = %.0f ms"%(1000/30,1000/10,1000/6))
for t in [0.2,0.4,0.6,1.0]:
    for f in [15,5,3]:
        N=t*f
        print("arrival t=%.1f s f=%d: N=%.1f periods -> dv/v< %.1f%%"%(t,f,N,100/(2*N)))
# FD cost: grid 70x70 at 10 m; OpenFWI uses dt=1 ms, 1000 steps; finer grid 5 m dx check CFL: dt <= dx/(vmax*sqrt2)*0.? 
nx=70; nz=70; nt=1000
print("cell updates per forward sim (10 m grid, 1000 steps):",nx*nz*nt)
# with absorbing pad of 20 cells each side approx
print("with 20-cell PML pad:",(nx+40)*(nz+20)*nt)
# 5 m grid with dt 0.5 ms
print("5 m grid 140x140x2000:",140*140*2000)
# CFL 2nd order 2D: v dt/dx <= 1/sqrt2 =0.707 ; 4500*0.001/10 = 0.45
print("Courant at 10 m, 1 ms, 4500 m/s: %.2f"%(4500*0.001/10))
# grid points per min wavelength at fmax 37.5 Hz v=1500: 
print("points per min wavelength (1500/40Hz)/10m = %.1f"%(1500/40/10))
