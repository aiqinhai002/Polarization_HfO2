#!python
#
from pymatgen.core.structure import Structure
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.io.vasp.sets import MPRelaxSet, MPStaticSet
from pymatgen.io.vasp.inputs import Incar
from pymatgen.io.vasp.outputs import Oszicar, Outcar
from pymatgen.core.units import Charge, Length
#from pymatgen.core.surface import Slab, SlabGenerator
#from itertools import combinations, product
import os
import shutil
import copy
import operator as op
import numpy as np
import itertools
from monty.serialization import loadfn

def polar_cal_berry_phase(in_file, base_file, N=21, exten_N=3, geom=True, polar_dir='c',
    vasp_run=False, run_str=None):
    '''
    Params
    ------
        in_file: file-like str
            The polarization POSCAR
        base_file: file-like str
            The central symmetrical POSCAR
        N: int
            The number of points between polarization and inverse polarization
        exten_N: int
            The number of points beyond the polarization (total: 2xN)
        geom: bool
            If run the geom opt before calculate the polarization
        polar_dir: 'a', 'b' or 'c'
            The direction of polarization
        vasp_run: bool
            If run the vasp, if false, only generate the files
        run_str: str
            The command for run vasp
    Returns
    -------
    Workflow:
        1. Geom opt the structure (polared)
        2. Generate the structure and input files
        3. Static calculation
    '''
    user_potcar_settings = {'Ba': 'Ba_sv', 'Mg': 'Mg', 'Nb': 'Nb_sv', 'Ti': 'Ti_sv', 'Zr': 'Zr_sv', 'Pb': 'Pb_d', 'O': 'O'}
    user_kpoints_settings = {'grid_density': 12000}
    user_potcar_functional = 'LDA'
    #user_incar_settings = {'NSW': 100, 'NELM': 200, 'ISPIN': 1, 'EDIFF': 1E-8, 'EDIFFG': 1E-4}
    eA = Charge(1., 'e') / Length(1., 'ang') ** 2
    C_per_m2 = float(eA.to('C m^-2'))
    if run_str is None:
        run_str = 'mpirun -np 8 vasp_std >> FP_OUT'
    struct = Structure.from_file(in_file, sort=True)
    struct_base = Structure.from_file(base_file, sort=True)
    #struct_base.to(filename=base_file, fmt='poscar')
    #struct_base = Structure.from_file(base_file)
    frac_coords_base = struct_base.frac_coords
    species_base = struct_base.species
    sites_base = struct_base.sites
    if geom:
        creat_folder('GeomOpt')
        os.chdir('GeomOpt')
        relax_set = MPRelaxSet(struct, user_incar_settings={'NSW': 100, 'NELM': 60, 'ISPIN': 1,
                     'EDIFF': 1E-6, 'EDIFFG': -1E-3},
                     user_kpoints_settings=user_kpoints_settings,
                     user_potcar_settings=user_potcar_settings,
                     user_potcar_functional=user_potcar_functional)
        relax_set.write_input('.')
        #Run
        if vasp_run:
            os.system(run_str)
            struct_geom = Structure.from_file('CONTCAR', sort=True)
        else:
            struct_geom = copy.deepcopy(struct)
        os.chdir('..')
    else:
        struct_geom = copy.deepcopy(struct)
    #print(struct_geom)
    species_geom = struct_geom.species
    sites_geom = struct_geom.sites
    frac_coords_geom = struct_geom.frac_coords
    lat_mat = struct_geom.lattice
    volume = lat_mat.volume
    axis_index = {'a': 0, 'b': 1, 'c': 2}

    a_polar = lat_mat.matrix[axis_index[polar_dir]]
    len_a_polar = np.linalg.norm(a_polar) 
    polar_delta = len_a_polar * C_per_m2 / volume

    #Get the proper frac_coords_base(by find the closest atom)
    frac_coords_base_new = []
    for i, frac_coord_geom in enumerate(frac_coords_geom):
        specie_geom = species_geom[i]
        disij = []
        frac_coord2s = []
        for j, frac_coord_base in enumerate(frac_coords_base):
            specie_base = species_base[j]
            if specie_geom == specie_base:
                (min_dis, frac_coord2_new) = cal_distance(frac_coord_geom, frac_coord_base, lat_mat.matrix)
                disij.append(min_dis)
                frac_coord2s.append(frac_coord2_new)
        min_index = np.argmin(np.array(disij))
        frac_coords_base_new.append(frac_coord2s[min_index])
    frac_coords_base_new = np.array(frac_coords_base_new)
    
    frac_coords_delta = frac_coords_geom - frac_coords_base_new
    '''
    tor_delta = 0.5
    for i, itemi in enumerate(frac_coords_delta):
        for j, itemj in enumerate(itemi):
            if itemj > tor_delta:
                frac_coords_delta[i, j] = frac_coords_delta[i, j] - 1
            elif itemj < -tor_delta:
                frac_coords_delta[i, j] = frac_coords_delta[i, j] + 1
    '''
    frac_coords_delta = -frac_coords_delta / float(N-1.) * 2.
    frac_coords_begin = frac_coords_geom - exten_N * frac_coords_delta

    #centroid_list = cal_o_centroid(struct_geom, center_atoms, around_atom=around_atom, num_atom=num_atom)
    creat_folder('Static')
    os.chdir('Static')
    energies = []
    polars = []
    for i in range(N+exten_N*2):
        creat_folder(str(i))
        os.chdir(str(i))
        static_set = MPStaticSet(struct_geom, lcalcpol=True,
            user_incar_settings={'DIPOL': [0.5, 0.5, 0.5], 'NELM': 200, 'EDIFF': 1E-7, 'ISPIN': 1},
            user_kpoints_settings=user_kpoints_settings,
            user_potcar_settings=user_potcar_settings,
            user_potcar_functional=user_potcar_functional)
        static_set.write_input('.')
        frac_coords_i = frac_coords_begin + float(i) * frac_coords_delta
        #struct_i = Structure.from_file('POSCAR')
        struct_i = copy.deepcopy(struct_geom)
        sites = struct_i.sites
        for i_site, site in enumerate(sites):
            site.frac_coords = frac_coords_i[i_site]
        struct_i.to(filename='POSCAR', fmt='poscar')
        if vasp_run:
            os.system(run_str)
            energy = Oszicar('OSZICAR').final_energy
            energies.append(energy)

            try:
                outcar = Outcar('OUTCAR')
                p_elec = outcar.p_elec / volume * C_per_m2
                p_ion = outcar.p_ion / volume * C_per_m2
            except Exception as e:
                p_elec = np.array([0., 0., 0.])
                p_ion = np.array([0., 0., 0.])
            
            p_all = (p_elec + p_ion)
            polars.append(np.hstack((p_elec, p_ion, p_all, polar_delta)))
        os.chdir('..')
    os.chdir('..')
    np.savetxt('ENERGY.TXT', energies)
    np.savetxt('POLAR-tmp.TXT', polars)
    if vasp_run:
        polars = np.array(polars)
        energies = np.array(energies)
        energies = np.reshape(energies, (len(energies), -1))
        p_sort = copy.deepcopy(polars[:, 6 + axis_index[polar_dir]])

        factor_delta = 0.5
        p_sort = sort_polar(p_sort, polar_delta, factor_delta=factor_delta)
        polars = np.hstack((polars, p_sort, energies, energies-min(energies)))
        np.savetxt('POLAR.TXT', polars)

def get_polar(path, N=27):
    result = []
    for i in range(N):
        path_abs = os.path.join(path, str(i), 'OUTCAR')
        outcar = Outcar(path_abs)
        p_elec = outcar.p_elec
        p_ion = outcar.p_ion
        p_all = p_elec + p_ion
        result.append(np.hstack((p_elec, p_ion, p_all)))
    np.savetxt('DipoleMoment-BT.TXT', result)

def sort_polar(p_sort, polar_delta, factor_delta=0.5):
    #factor_delta = 0.5
    N_p = len(p_sort)
    for i in range(1, N_p):
        polar_deltai = p_sort[i] - p_sort[i-1]
        if abs(polar_deltai) > factor_delta*polar_delta:
            N_delta = round(polar_deltai/polar_delta)
            p_sort[i] = p_sort[i] - float(N_delta) * polar_delta
    if N_p % 2:
        p_center = p_sort[int((N_p - 1)/2)]
    else:
        p_center = (p_sort[int(N_p/2)] + p_sort[int(N_p/2)-1])/2
    N_move = round(p_center/polar_delta)
    p_sort = np.array(p_sort) - N_move * polar_delta
    p_sort = np.reshape(p_sort, (N_p, -1))
    return p_sort

def optimiz_with_pymatgen(in_file, isifs=['7'], vasp_run=True, run_str=None):
    if run_str is None:
        run_str = 'mpirun -np 28 vasp_std >> FP_OUT'
    struct = Structure.from_file(in_file)
    user_potcar_settings = {'Ba': 'Ba_sv', 'Mg': 'Mg', 'Nb': 'Nb_sv', 'Ti': 'Ti_sv', 'Zr': 'Zr_sv', 'Pb': 'Pb_d', 'O': 'O'}
    user_kpoints_settings = {'grid_density': 3000}
    user_potcar_functional = 'LDA'
    user_incar_settings={'NSW': 100, 'NELM': 60, 'ISPIN': 1, 'EDIFF': 1E-6, 'EDIFFG': -1E-3, 
        'ISMEAR': 0, 'SIGMA': 0.1, 'AMIN': 0.01, 'NPAR': 4, 'ALGO': 'Normal'}
    for isif in isifs:
        try:
            struct = copy.deepcopy(struct_geom)
        except Exception as e:
            pass
        isif_foler = 'ISIF{}'.format(isif)
        creat_folder(isif_foler)
        os.chdir(isif_foler)
        user_incar_settings.update({'ISIF': isif})
        relax_set = MPRelaxSet(struct, user_incar_settings=user_incar_settings,
                user_kpoints_settings=user_kpoints_settings,
                user_potcar_settings=user_potcar_settings,
                user_potcar_functional=user_potcar_functional)
        relax_set.write_input('.')
        #Run
        if vasp_run:
            os.system(run_str)
            struct_geom = Structure.from_file('CONTCAR', sort=True)
        else:
            struct_geom = copy.deepcopy(struct)
        os.chdir('..')
    user_incar_settings.update({'ISMEAR': -5, 'NSW': 0, 'ISIF': 2, 'ALGO': 'Fast'})
    #user_incar_settings.pop('EDIFFG')
    creat_folder('Static')
    os.chdir('Static')
    static_set = MPStaticSet(struct_geom, lcalcpol=True,
        user_incar_settings=user_incar_settings,
        user_kpoints_settings=user_kpoints_settings,
        user_potcar_settings=user_potcar_settings,
        user_potcar_functional=user_potcar_functional)
    static_set.write_input('.')
    if vasp_run:
        os.system(run_str)
    os.chdir('..')

def set_atom_sequence(struct, atom_seq, rep_atoms, axis='b'):
    n_seq = len(atom_seq)
    if axis.lower() == 'a':
        ax_ind = 0
    elif axis.lower() == 'b':
        ax_ind = 1
    else:
        ax_ind = 2
    sort_ax_val = []
    sites = struct.sites
    for site in sites:
        frac_coords = site.frac_coords
        sort_ax_val.append(frac_coords[ax_ind])
    sites = sort_x_by_y(sites, sort_ax_val)
    i = 0
    for site in sites:
        specie = site.specie
        if specie.symbol in rep_atoms:
            j = i % n_seq
            site.species = atom_seq[j]
            i += 1
    return Structure.from_sites(sites)

    '''
    Calculate the centroid of Oxygen octahedron

    '''
    all_around = np.array([[0., 0., 0.], [1., 0., 0.], [-1., 0., 0.], 
                                         [0., 1., 0.], [0., -1., 0.], 
                                         [0., 0., 1.], [0., 0., -1.],
                                         [1., 1., 0.], [-1., -1., 0.],
                                         [1., 0., 1.], [-1., 0., -1.],
                                         [0., 1., 1.], [0., -1., -1.],
                                         [1., -1., 0.], [-1., 1., 0.],
                                         [1., 0., -1.], [-1., 0., 1.],
                                         [0., 1., -1.], [0., -1., 1.],
                                         [1., 1., 1.], [-1., -1., -1.],
                                         [1., 1., -1.], [-1., -1., 1.],
                                         [1., -1., 1.], [-1., 1., -1.],
                                         [-1., 1., 1.], [1., -1., -1.]])
    lat_mat = struct.lattice.matrix
    print(lat_mat)
    sites = struct.sites
    sites_o = copy.deepcopy(sites)
    centroid_list = []
    for i, site in enumerate(sites):
        specie = site.specie
        frac_coords = site.frac_coords
        #for each center atom
        min_vals = []
        min_o_coords = []
        for j, site_o in enumerate(sites_o):
            if i != j:
                frac_coords_o = site_o.frac_coords
                #For each o atom
                #around + o atom - center atom
                frac_coords_o = all_around + np.array([frac_coords_o]) - np.array([frac_coords])
                cart_coords_o = np.dot(frac_coords_o, lat_mat)
                dis_bo = np.linalg.norm(cart_coords_o, axis=1)
                min_index = np.argmin(dis_bo)
                min_val = dis_bo[min_index]
                min_o_coord = cart_coords_o[min_index]
                min_vals.append(min_val)
                min_o_coords.append(min_o_coord)
        min_o_coords = sort_x_by_y(min_o_coords, min_vals)
        min_vals = sorted(min_vals)
        min_o_coords = min_o_coords[0:num_atom]
        min_vals = min_vals[0:num_atom]
        centroid_coords = np.sum(np.array(min_o_coords), axis=0) / float(num_atom)
        centroid_coords_frac = np.dot(centroid_coords, np.linalg.inv(lat_mat)) + np.array([frac_coords])
        centroid_list.append({'Ele': specie.symbol, 'Batom': frac_coords, 'Ocenter': centroid_coords_frac[0], 
            'Polar': frac_coords - centroid_coords_frac[0], 'LatVec': lat_mat,
            'Distance': min_vals})   

    result_list = []
    for item in centroid_list:
        result_list.append([item['Ele']] + item['Batom'].tolist() + item['Distance'])
        print([item['Ele']] + item['Batom'].tolist() + item['Distance'])
    #print(result_list)
    return(centroid_list)   
