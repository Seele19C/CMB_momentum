# import head
from nbodykit.lab import *
from nbodykit import style, setup_logging
import matplotlib.pyplot as plt
plt.style.use(style.notebook)
import numpy as np
import nbodykit as nb
from joblib import dump, load
import os


#const
LOS=[1,0,0]
z_eff=0.59339

#cos
cosmo = cosmology.Planck15
H=cosmo.efunc(z_eff)*100
a=1/(1+z_eff)

#类
struct_dtype = np.dtype([
    ('Pos', np.float32, 3),
    ('Vel', np.float32, 3),
    ('Vmax', np.float32),
    ('CentralMvir', np.float32),
    ('StellarMass', np.float32),
    ('HImass', np.float32)
])

# rp = np.dtype([
#     ('r00', np.object),
#     ('r01', np.object),
#     ('r02', np.object),
#     ('P00', np.object),
#     ('P01', np.object),
#     ('P02', np.object),
# 
#读数据
def data_read(num):
    data = np.fromfile(f'../filtered_data_h_{num}.dat', dtype=struct_dtype)
    cat = nb.source.catalog.ArrayCatalog(data)
    return cat
#加红移
def RED(cat):
    line_of_sight_x=(1,0,0)
    cat['vel_x']=cat['Vel']*line_of_sight_x/(a*H)
    cat['RsdPos_x']=cat['Pos']+(cat['vel_x']*line_of_sight_x)
    cat['Vx'] = cat['Vel'][:,0]
    return cat
#加权重
def weight(cat):
    # total_mvir = cat['CentralMvir'].compute().sum()
    mean_mvir = cat['CentralMvir'].compute().mean()
    central_mvir_values = cat['CentralMvir'].compute()
    # weights = central_mvir_values / (total_mvir/83492032)
    weights = central_mvir_values / mean_mvir
    cat['weight'] = weights
    return cat
#按gamma加权
def weight_gamma(cat, gamma):
    cat['CentralMvir_gamma'] = cat['CentralMvir']**gamma
    mean_mvir = cat['CentralMvir_gamma'].compute().mean()
    central_mvir_values = cat['CentralMvir_gamma'].compute()
    weights = (central_mvir_values) / mean_mvir
    cat['weight'] = weights
    return cat

#检测有无加权及mesh和momentum计算
# def test_m(cat):
#         if 'weight' not in cat:
#         #无权重
#         cat['weight'] = 1

#     if 'RsdPos_x' in cat:
#         #有红移
#         LOS=[1,0,0]
#         momentum_mesh = cat.to_mesh(interlaced=True,compensated=True, Nmesh=512, BoxSize=1000, window='tsc', position='RsdPos_x', value='Vx',weight='weight')
#         mesh = cat.to_mesh(interlaced=True,compensated=True, Nmesh=512, BoxSize=1000, window='tsc', position='RsdPos_x')
#     else:
#         #无红移
#         momentum_mesh = cat.to_mesh(interlaced=True,compensated=True, Nmesh=512, BoxSize=1000, window='tsc', position='Pos', value='Vx',weight='weight')
#         mesh = cat.to_mesh(interlaced=True,compensated=True, Nmesh=512, BoxSize=1000, window='tsc', position='Pos')
#     return momentum_mesh, mesh


#计算momentum
def momentum(cat):
    if 'weight' not in cat:
        #无权重
        cat['weight'] = 1

    if 'RsdPos_x' in cat:
        #有红移
        LOS=(1,0,0)
        cat['RsdPos_x'] = cat['RsdPos_x']%1000
        momentum_mesh = cat.to_mesh(interlaced=True,compensated=True, Nmesh=256, BoxSize=1000, window='tsc', position='RsdPos_x', value='Vx',weight='weight')
        mesh = cat.to_mesh(interlaced=True,compensated=True, Nmesh=256, BoxSize=1000, window='tsc', position='RsdPos_x')
    else:
        #无红移
        momentum_mesh = cat.to_mesh(interlaced=True,compensated=True, Nmesh=256, BoxSize=1000, window='tsc', position='Pos', value='Vx',weight='weight')
        mesh = cat.to_mesh(interlaced=True,compensated=True, Nmesh=256, BoxSize=1000, window='tsc', position='Pos')

    # r_p = np.zeros((), dtype=rp)
    # r_p['r00'] = FFTPower(mesh, mode='1d', los=[1,0,0])
    # r_p['r01'] = FFTPower(momentum_mesh, mode='1d', poles=[1], second=mesh, los=LOS)
    # r_p['r11'] = FFTPower(momentum_mesh, mode='1d', poles=[2], los=LOS)
    # r_p['r00'] = r_p['r00'].power['power'].real - r_p['r00'].attrs['shotnoise']
    # r_p['r01'] = 2 * r_p['r01'].poles['k'] * r_p['r01'].poles['power_1'].imag
    # r_p['r11'] = 1.5 * r_p['r11'].poles['k']**2 * r_p['r11'].poles['power_2'].real
    r00 = FFTPower(mesh, mode='1d', los=[1,0,0])
    r01 = FFTPower(momentum_mesh, mode='1d', poles=[1], second=mesh, los=LOS)
    r11 = FFTPower(momentum_mesh, mode='1d', poles=[2], los=LOS)
    P00 = r00.power['power'].real - r00.attrs['shotnoise']
    P01 = 2 * r01.poles['k'] * r01.poles['power_1'].imag
    P11 = 1.5 * r11.poles['k']**2 * r11.poles['power_2'].real
    return r00, r01, r11, P00, P01, P11



#轻量版
def momentum_01(cat, num, gamma=None, cache=True):
    if 'weight' not in cat:
        #无权重
        cat['weight'] = 1
    if 'RsdPos_x' in cat:
        #有红移
        LOS=(1,0,0)
        cat['RsdPos_x'] = cat['RsdPos_x']%1000
        momentum_mesh = cat.to_mesh(interlaced=True,compensated=True, Nmesh=512, BoxSize=1000, window='tsc', position='RsdPos_x', value='Vx',weight='weight')
        mesh = cat.to_mesh(interlaced=True,compensated=True, Nmesh=512, BoxSize=1000, window='tsc', position='RsdPos_x')
    else:
        #无红移
        momentum_mesh = cat.to_mesh(interlaced=True,compensated=True, Nmesh=512, BoxSize=1000, window='tsc', position='Pos', value='Vx',weight='weight')
        mesh = cat.to_mesh(interlaced=True,compensated=True, Nmesh=512, BoxSize=1000, window='tsc', position='Pos')
    r01 = FFTPower(momentum_mesh, mode='1d', poles=[1], second=mesh, los=LOS)
    P01 = 2 * r01.poles['k'] * r01.poles['power_1'].imag
    #cache
    if cache is not False: 
        cache_write(num=num, r01=r01, P01=P01, gamma=gamma)
    return r01, P01
#00
def momentum_00(cat, num, gamma=None, cache=True):
    if 'weight' not in cat:
        #无权重
        cat['weight'] = 1
    if 'RsdPos_x' in cat:
        #有红移
        LOS=(1,0,0)
        cat['RsdPos_x'] = cat['RsdPos_x']%1000
        # momentum_mesh = cat.to_mesh(interlaced=True,compensated=False, Nmesh=256, BoxSize=1000, window='tsc', position='RsdPos_x', value='Vx',weight='weight')
        mesh = cat.to_mesh(interlaced=True,compensated=True, Nmesh=512, BoxSize=1000, window='tsc', position='RsdPos_x')
    else:
        #无红移
        # momentum_mesh = cat.to_mesh(interlaced=False,compensated=False, Nmesh=256, BoxSize=1000, window='tsc', position='Pos', value='Vx',weight='weight')
        mesh = cat.to_mesh(interlaced=True,compensated=True, Nmesh=512, BoxSize=1000, window='tsc', position='Pos')
    r00 = FFTPower(mesh, mode='1d', los=[1,0,0])
    P00 = r00.power['power'].real - r00.attrs['shotnoise']
    #cache
    # if cache is not False: 
    #     cache_write(num=num, r01=r01, P01=P01, gamma=gamma)
    return r00, P00
#compensate=False
def momentum_00_nocompensate(cat, num, gamma=None, cache=True):
    if 'weight' not in cat:
        #无权重
        cat['weight'] = 1
    if 'RsdPos_x' in cat:
        #有红移
        LOS=(1,0,0)
        cat['RsdPos_x'] = cat['RsdPos_x']%1000
        # momentum_mesh = cat.to_mesh(interlaced=True,compensated=False, Nmesh=256, BoxSize=1000, window='tsc', position='RsdPos_x', value='Vx',weight='weight')
        mesh = cat.to_mesh(interlaced=True,compensated=False, Nmesh=256, BoxSize=1000, window='tsc', position='RsdPos_x')
    else:
        #无红移
        # momentum_mesh = cat.to_mesh(interlaced=False,compensated=False, Nmesh=256, BoxSize=1000, window='tsc', position='Pos', value='Vx',weight='weight')
        mesh = cat.to_mesh(interlaced=True,compensated=False, Nmesh=256, BoxSize=1000, window='tsc', position='Pos')
    r00 = FFTPower(mesh, mode='1d', los=[1,0,0])
    P00 = r00.power['power'].real - r00.attrs['shotnoise']
    #cache
    # if cache is not False: 
    #     cache_write(num=num, r01=r01, P01=P01, gamma=gamma)
    return r00, P00
#写cache
def cache_write(num, r01, P01, gamma=None):
    os.makedirs('./momentum_cache', exist_ok=True)
    num_str = str(num).replace('.', '_')
    gamma_str = str(gamma).replace('.', '_')
    if gamma is not None:
        dump(r01, './momentum_cache/r01_cache_%s_%s.joblib'%(num_str,gamma_str))
        dump(P01, './momentum_cache/P01_cache_%s_%s.joblib'%(num_str,gamma_str))
        print('./momentum_cache/P01_cache_%s_%s.joblib'%(num_str,gamma_str))
        print('./momentum_cache/P01_cache_%s_%s.joblib'%(num_str,gamma_str))
    elif gamma is None:
        dump(r01, './momentum_cache/r01_cache_%s_noweight.joblib'%(num_str))
        dump(P01, './momentum_cache/P01_cache_%s_noweight.joblib'%(num_str))
        print('./momentum_cache/r01_cache_%s_noweight.joblib'%(num_str))
        print('./momentum_cache/P01_cache_%s_noweight.joblib'%(num_str))

#读cache
def cache_read(num, gamma=None):
    num_str = str(num).replace('.', '_')
    gamma_str = str(gamma).replace('.', '_')
    if gamma is not None:
        r01 = load('./momentum_cache/r01_cache_%s_%s.joblib'%(num_str,gamma_str))
        P01 = load('./momentum_cache/P01_cache_%s_%s.joblib'%(num_str,gamma_str))
    elif gamma is None:
        r01 = load('./momentum_cache/r01_cache_%s_noweight.joblib'%(num_str))
        P01 = load('./momentum_cache/P01_cache_%s_noweight.joblib'%(num_str))
    return r01, P01
#density
# def density(cat):
#     LOS=[1,0,0]
#     mesh = cat.to_mesh(interlaced=True,compensated=True, Nmesh=512, BoxSize=1000, window='tsc', position='RsdPos_x')
#     r00 = FFTPower(mesh, mode='1d', los=[1,0,0])
#     r01 = FFTPower(mesh, mode='1d', poles=[1], los=LOS)
#     r11 = FFTPower(mesh, mode='1d', poles=[2], los=LOS)
#     P00 = r00.power['power'].real - r00.attrs['shotnoise']
#     P01 = 2 * r01.poles['k'] * r01.poles['power_1'].imag
#     P11 = 1.5 * r11.poles['k']**2 * r11.poles['power_2'].real
#     return r00, r01, r11, P00, P01, P11
# def density_pole(cat):
#     LOS=[1,0,0]
#     mesh = cat.to_mesh(window='tsc', Nmesh=512, BoxSize=1000,interlaced=True,compensated=True, position='RsdPos_x')
#     r = FFTPower(mesh, mode='2d', dk=0.005, kmin=0.01, Nmu=5, los=[1,0,0], poles=[0,2,4])
#     poles = r.poles
#     # for ell in [0, 2, 4]:
#     # label = r'$\ell=%d$' % (ell)
#     ell_2 = 2
#     P_2 = poles['power_%d' %ell_2].real
#     # if ell == 0: P = P - poles.attrs['shotnoise']
#     ell_4 = 4
#     P_4 = poles['power_%d' %ell_4].real
#     return r
def density(cat):
    
    if 'weight' not in cat:
        #无权重
        cat['weight'] = 1

    if 'RsdPos_x' in cat:
        cat['RsdPos_x'] = cat['RsdPos_x']%1000
        mesh = cat.to_mesh(window ='tsc', Nmesh=256,BoxSize=1000,compensated=False,interlaced=True, position='RsdPos_x')
        r = FFTPower(mesh, mode='2d', dk=0.005, kmin=0.01, Nmu=5, los=[1,0,0], poles=[0,2,4])
    # else:
    #     mesh = cat.to_mesh(window='tsc', Nmesh=512, BoxSize=1000,interlaced=True,compensated=True, position='Pos', weight='weight')
    #     r = FFTPower(mesh, mode='2d', dk=0.005, kmin=0.01, Nmu=5, los=[1,0,0], poles=[2,4])     
    poles = r.poles
    print(poles)
    print("variables = ", poles.variables)
    for ell in [0, 2, 4]:
        label = r'$\ell=%d$' % (ell)
        P = poles['power_%d' %ell].real
        if ell == 0: P = P - poles.attrs['shotnoise']
        # if ell == 2
        # line, = plt.plot(x, y, label="")
        plt.plot(poles['k'], poles['k'] * P, label=label)
        # line.set_color('')
    plt.legend(loc=0)
    plt.xlabel(r"$k$ [$h \ \mathrm{Mpc}^{-1}$]")
    plt.ylabel(r"$k \ P_\ell$ [$h^{-2} \mathrm{Mpc}^2$]")


def test(cat):
    if 'RsdPos_x' in cat:
        print("列 'RsdPos_x' 存在")
    else:
        print("列 'RsdPos_x' 不存在")

#测试版


def correlation_11(cat_1, cat_2):
    LOS=(1,0,0)
    cat_1['RsdPos_x'] = cat_1['RsdPos_x']%1000
    cat_2['RsdPos_x'] = cat_2['RsdPos_x']%1000
    mesh_1 = cat_1.to_mesh(interlaced=True,compensated=True, Nmesh=512, BoxSize=1000, window='tsc', position='RsdPos_x')
    momentum_mesh_2 = cat_2.to_mesh(interlaced=True,compensated=True, Nmesh=512, BoxSize=1000, window='tsc', position='RsdPos_x', value='Vx',weight='weight')
    r01 = FFTPower(momentum_mesh_2, mode='1d', poles=[1], second=mesh_1, los=LOS)
    P01 = 2 * r01.poles['k'] * r01.poles['power_1'].imag
    return r01, P01