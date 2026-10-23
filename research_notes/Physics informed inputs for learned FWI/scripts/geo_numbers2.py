import numpy as np
print("moveout and sensitivity to 10% velocity error, far offset")
for z,v,x in [(200,2000,700),(200,2000,350),(400,2500,700),(700,3000,700),(700,3000,350),(700,1500,700)]:
    t0=2*z/v; tx=np.sqrt(t0**2+(x/v)**2); dmo=tx-t0; sens=x**2/(v**2*tx)*0.1
    print("z=%d v=%d x=%d t0=%.3f tx=%.3f moveout=%.0f ms, 10%%dv -> %.0f ms RMO (%.2f periods @15Hz)"%(z,v,x,t0,tx,dmo*1e3,sens*1e3,sens*15))
print("fold vs midpoint, shots at 0,170,345,520,690, receivers 0..690")
shots=[0,170,345,520,690]; rec=np.arange(0,700,10)
for m in [25,50,100,150,250,345,450,600,650,690]:
    f=sum(1 for s in shots if (2*m-s)>=0 and (2*m-s)<=690)
    print(m,f)
print("acoustic R(0) examples")
for v1,v2 in [(1500,4500),(2000,2200),(3000,2500),(2500,3000),(1600,4300)]:
    print(v1,v2,"R0=%.3f two-way T=1-R^2=%.3f thetac=%s"%((v2-v1)/(v2+v1),1-((v2-v1)/(v2+v1))**2, "%.1f"%np.degrees(np.arcsin(v1/v2)) if v2>v1 else "none"))
print("source wavefield storage per shot float32 70x70x1000 MB:",70*70*1000*4/1e6)
