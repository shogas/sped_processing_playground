import math

import matplotlib
matplotlib.use('Qt5Agg')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button, TextBox

from transforms3d.euler import euler2mat
from transforms3d.euler import mat2euler
from transforms3d.euler import axangle2euler
from transforms3d.euler import axangle2mat

import pyxem as pxm
from pyxem.libraries.structure_library import StructureLibrary
from pyxem.generators.indexation_generator import IndexationGenerator

import diffpy.structure

structure_zb_file = 'D:/Dokumenter/MTNANO/Prosjektoppgave/Data/Gen/NN_test_data/GaAs_mp-2534_conventional_standard.cif'
# structure_zb_file = 'D:/Dokumenter/MTNANO/Prosjektoppgave/Data/Gen/NN_test_data/GaAs_mp-2534_primitive.cif'
structure_wz_file = 'D:/Dokumenter/MTNANO/Prosjektoppgave/Data/Gen/NN_test_data/GaAs_mp-8883_conventional_standard.cif'
# structure_wz_file = 'D:/Dokumenter/MTNANO/Prosjektoppgave/Data/Gen/NN_test_data/GaAs_mp-8883_primitive.cif'

beam_energy_keV = 200
specimen_thickness = 80  # Ångström
target_pattern_dimension_pixels = 144
angstrom_per_pixel = 19.6
half_pattern_size = target_pattern_dimension_pixels // 2
simulated_gaussian_sigma = 0.02
reciprocal_angstrom_per_pixel = 0.035 # From 110 direction, compared to a_crop

# For local rotation list
max_theta = 45
resolution = 10


def structure_manual():
    # This seems to give the same result as structure_zb_file
    a = 5.75
    lattice = diffpy.structure.lattice.Lattice(a, a, a, 90, 90, 90)
    atom_list = []
    for x, y, z in [(0, 0, 0), (0.5, 0, 0.5), (0, 0.5, 0.5), (0.5, 0.5, 0)]:
        atom_list.append(diffpy.structure.atom.Atom(atype='Ga', xyz=[x,      y,      z],      lattice=lattice))
        atom_list.append(diffpy.structure.atom.Atom(atype='As', xyz=[x+0.25, y+0.25, z+0.25], lattice=lattice))
    return diffpy.structure.Structure(atoms=atom_list, lattice=lattice)


structure_zb = diffpy.structure.loadStructure('file:///' + structure_zb_file)
structure_wz = diffpy.structure.loadStructure('file:///' + structure_wz_file)

structures = [
        {
            'name': 'ZB',
            'structure': structure_zb,
            'corners': [ (0, 0, 1), (1, 0, 1), (1, 1, 1) ]
        },
        {
            'name': 'WZ',
            'structure': structure_wz,
            'corners': [ (0, 0, 0, 1), (1, 1, -2, 0), (1, 0, -1, 0) ]
        }
]
current_structure = 0



def angle_between_cartesian(a, b):
    return math.acos(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)))


def angle_between_directions(structure,
                            direction1,
                            direction2):
    """Returns the angle in radians between two crystal directions in the given structure."""

    a = structure.lattice.a
    b = structure.lattice.b
    c = structure.lattice.c
    alpha = np.deg2rad(structure.lattice.alpha)
    beta = np.deg2rad(structure.lattice.beta)
    gamma = np.deg2rad(structure.lattice.gamma)

    u1 = direction1[0]
    v1 = direction1[1]
    w1 = direction1[2]

    u2 = direction2[0]
    v2 = direction2[1]
    w2 = direction2[2]

    L = a**2*u1*u2 + b**2*v1*v2 + c**2*w1*w2 \
        + b*c*(v1*w2 + w1*v2)*math.cos(alpha) \
        + a*c*(w1*u2 + u1*w2)*math.cos(beta) \
        + a*b*(u1*v2 + v1*u2)*math.cos(gamma)

    I1 = np.sqrt(a**2 * u1**2 + b**2*v1**2 + c**2*w1**2 \
        + 2*b*c*v1*w1*math.cos(alpha) \
        + 2*a*c*w1*u1*math.cos(beta) \
        + 2*a*b*u1*v1*math.cos(gamma))

    I2 = np.sqrt(a**2 * u2**2 + b**2*v2**2 + c**2*w2**2 \
        + 2*b*c*v2*w2*math.cos(alpha) \
        + 2*a*c*w2*u2*math.cos(beta) \
        + 2*a*b*u2*v2*math.cos(gamma))

    return math.acos(L/(I1*I2))


def plot_3d_axes(ax):
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    axis_a = ax.plot([0, 1], [0, 0], [0, 0], c=(1, 0, 0))
    axis_b = ax.plot([0, 0], [0, 1], [0, 0], c=(0, 1, 0))
    axis_c = ax.plot([0, 0], [0, 0], [0, 1], c=(0, 0, 1))

    rot_count = generate_rotation_list(structures[current_structure], 0, 0, 0, max_theta, resolution).shape[0]
    return axis_a, axis_b, axis_c, ax.scatter([0]*rot_count, [0]*rot_count, [0]*rot_count)


# TODO: Look at https://www.ctcms.nist.gov/%7Elanger/oof2man/RegisteredClass-Bunge.html
def generate_rotation_list(structure, phi, theta, psi, max_theta, resolution):
    # TODO: Symmetry considerations

    zone_to_rotation = np.identity(3)
    lattice_to_zone = np.identity(3)
    lattice_to_zone = euler2mat(phi, -theta, psi, 'rzxz')

    # This generates rotations around the given axis, with a denser sampling close to the axis
    max_theta = np.deg2rad(max_theta)
    resolution = np.deg2rad(resolution)
    min_psi = -np.pi
    max_psi = np.pi
    theta_count = math.ceil(max_theta / resolution)
    psi_count = math.ceil((max_psi - min_psi) / resolution)
    rotations = np.empty((theta_count, psi_count, 3, 3))
    for i, local_theta in enumerate(np.arange(0, max_theta, resolution)):
        for j, local_psi in enumerate(np.arange(min_psi, max_psi, resolution)):
            zone_to_rotation = euler2mat(0, local_theta, local_psi, 'sxyz')
            lattice_to_rotation = np.matmul(lattice_to_zone, zone_to_rotation)
            rotations[i, j] = lattice_to_rotation

    return rotations.reshape(-1, 3, 3)


def transformation_matrix_to_cartesian(structure):
    # From formula for change of basis, see hand written description
    a = structure.lattice.a
    b = structure.lattice.b
    c = structure.lattice.c
    alpha = np.deg2rad(structure.lattice.alpha)  # angle a to c
    beta  = np.deg2rad(structure.lattice.beta)    # angle b to c
    gamma = np.deg2rad(structure.lattice.gamma)  # angle a to b

    cos_alpha = math.cos(alpha)
    cos_beta = math.cos(beta)
    cos_gamma = math.cos(gamma)
    sin_gamma = math.sin(gamma)


    factor_e3_0 = cos_beta
    factor_e3_1 = (cos_alpha - cos_beta*cos_gamma)/sin_gamma
    factor_e3_2 = math.sqrt(1 - np.dot(factor_e3_0, factor_e3_0) - np.dot(factor_e3_1, factor_e3_1))

    # Columns are the transformations of the corresponding cartesian basis vector.
    return np.array([
        [a, b*cos_gamma, c*factor_e3_0],
        [0, b*sin_gamma, c*factor_e3_1],
        [0, 0,           c*factor_e3_2]
    ])


def direction_to_cartesian(structure_from, direction_from):
    transform = transformation_matrix_to_cartesian(structure_from)
    return np.dot(transform, direction_from)


def direction_from_cartesian(structure_to, direction_to):
    transform = np.linalg.inv(transformation_matrix_to_cartesian(structure_to))
    return np.dot(transform, direction_to)


def generate_complete_rotation_list(structure, corner_a, corner_b, corner_c, inplane_rotations, resolution):
    """Generate a rotation list covering the inverse pole figure specified by three
        corners in cartesian coordinates.

    Arguments:
        structure: diffpy.structure.Structure, used for calculating angles
        corner_a, corner_b, corner_c: The three corners of the inverse pole
            figure given by three coordinates in the coordinate system
            specified by the structure lattice (crystal directions).
        resolution: Angular resolution in radians of the generated rotation
            list.
        inplane_rotations: List of angles in radians for in-plane rotation
            of the diffraction pattern. This corresponds to the third Euler
            angle rotation. The rotation list will be generated for each of
            these angles, and combined. This should be done automatically, but
            by including all possible rotations in the rotation list, it
            becomes too large.

    Returns:
        Rotations covering the inverse pole figure given as a `np.array` of Euler
            angles in degress. This `np.array` can be passed directly to pyxem.
    """

    # Convert the crystal directions to cartesian vectors, and then normalize
    # to get the directions.
    if len(corner_a) == 4:
        corner_a = uvtw_to_uvw(*corner_a)
    if len(corner_b) == 4:
        corner_b = uvtw_to_uvw(*corner_b)
    if len(corner_c) == 4:
        corner_c = uvtw_to_uvw(*corner_c)

    corner_a = direction_to_cartesian(structure, corner_a)
    corner_b = direction_to_cartesian(structure, corner_b)
    corner_c = direction_to_cartesian(structure, corner_c)

    corner_a /= np.linalg.norm(corner_a)
    corner_b /= np.linalg.norm(corner_b)
    corner_c /= np.linalg.norm(corner_c)

    angle_a_to_b = angle_between_cartesian(corner_a, corner_b)
    angle_a_to_c = angle_between_cartesian(corner_a, corner_c)
    angle_b_to_c = angle_between_cartesian(corner_b, corner_c)
    axis_a_to_b = np.cross(corner_a, corner_b)
    axis_a_to_c = np.cross(corner_a, corner_c)

    # Input validation. The corners have to define a non-degenerate triangle
    if np.count_nonzero(axis_a_to_b) == 0:
        raise ValueError("Directions a and b are parallel")
    if np.count_nonzero(axis_a_to_c) == 0:
        raise ValueError("Directions a and c are parallel")

    # Find the maxiumum number of points we can generate, given by the
    # resolution, then allocate storage for them. For the theta direction,
    # ensure that we keep the resolution also along the direction to the corner
    # b or c farthest away from a.
    theta_count = math.ceil(max(angle_a_to_b, angle_a_to_c) / resolution)
    phi_count = math.ceil(angle_b_to_c / resolution)
    inplane_rotation_count = len(inplane_rotations)
    rotations = np.zeros((theta_count, phi_count, inplane_rotation_count, 3, 3))

    # Generate a list of theta_count evenly spaced angles theta_b in the range
    # [0, angle_a_to_b] and an equally long list of evenly spaced angles
    # theta_c in the range[0, angle_a_to_c].
    for i, (theta_b, theta_c) in enumerate(
            zip(np.linspace(0, angle_a_to_b, theta_count),
                np.linspace(0, angle_a_to_c, theta_count))):
        # Define the corner local_b at a rotation theta_b from corner_a toward
        # corner_b on the circle surface. Similarly, define the corner local_c
        # at a rotation theta_c from corner_a toward corner_c.

        rotation_a_to_b = axangle2mat(axis_a_to_b, theta_b)
        rotation_a_to_c = axangle2mat(axis_a_to_c, theta_c)
        local_b = np.dot(rotation_a_to_b, corner_a)
        local_c = np.dot(rotation_a_to_c, corner_a)

        # Then define an axis and a maximum rotation to create a great cicle
        # arc between local_b and local_c. Ensure that this is not a degenerate
        # case where local_b and local_c are coincident.
        angle_local_b_to_c = angle_between_cartesian(local_b, local_c)
        axis_local_b_to_c = np.cross(local_b, local_c)
        if np.count_nonzero(axis_local_b_to_c) == 0:
            # Theta rotation ended at the same position. First position, might
            # be other cases?
            axis_local_b_to_c = corner_a
        axis_local_b_to_c /= np.linalg.norm(axis_local_b_to_c)

        # Generate points along the great circle arc with a distance defined by
        # resolution.
        phi_count_local = math.ceil(angle_local_b_to_c / resolution)
        for j, phi in enumerate(
                np.linspace(0, angle_local_b_to_c, phi_count_local)):
            rotation_phi = axangle2mat(axis_local_b_to_c, phi)

            for k, psi in enumerate(inplane_rotations):
                rotation_psi = axangle2mat((0, 0, 1), psi)

                # Combine the rotations. Order is important. The matrix is
                # applied from the left, and we rotate by theta first toward
                # local_b, then across the triangle toward local_c, and finally
                # an inplane rotation psi
                rotations[i, j, k] = rotation_psi @ (rotation_phi @ rotation_a_to_b)

    return rotations


def rotation_matrices_to_euler(rotation_list):
    """Convert a rotation list in matrix form to Euler angles in degrees.

    Arguments:
        rotation_list: np.array of three or more dimensions, where the last two
            correspond the 3x3 matrix

    Returns:
        Rotation list in Euler angles in degrees with duplicates removed.
    """
    # Remove duplicates
    rotation_list = np.unique(rotation_list.reshape(-1, 3, 3), axis=0)
    # Convert to euler angles in degrees
    return np.rad2deg([mat2euler(rotation_matrix, 'rzxz') for rotation_matrix in rotation_list])


def update_rotation(rotation_matrices):
    rotation_matrices = rotation_matrices.reshape(-1, 3, 3)
    v = np.empty((rotation_matrices.shape[0], 3))
    for i, rotation_matrix in enumerate(rotation_matrices):
        v[i] = np.dot(rotation_matrix, np.array([0, 0, 1]).T)
    rotation_scatter._offsets3d = v.T


def update_pattern(_ = None):
    reciprocal_angstrom_per_pixel = slider_scale.val
    simulated_gaussian_sigma = slider_sigma.val

    phi   = np.deg2rad(slider_phi.val)
    theta = np.deg2rad(slider_theta.val)
    psi   = np.deg2rad(slider_psi.val)

    structure_rotation = euler2mat(phi, theta, psi, axes='rzxz')

    structure = structures[current_structure]['structure']
    lattice_rotated = diffpy.structure.lattice.Lattice(
            structure.lattice.a,
            structure.lattice.b,
            structure.lattice.c,
            structure.lattice.alpha,
            structure.lattice.beta,
            structure.lattice.gamma,
            baserot=structure_rotation)
    structure.placeInLattice(lattice_rotated)

    reciprocal_radius = reciprocal_angstrom_per_pixel*(half_pattern_size - 1)
    sim = gen.calculate_ed_data(structure, reciprocal_radius, with_direct_beam=False)
    # sim.intensities = np.full(sim.direct_beam_mask.shape, 1)  # For testing, ignore intensities
    # sim._intensities = np.log(1 + sim._intensities)
    s = sim.as_signal(target_pattern_dimension_pixels, simulated_gaussian_sigma, reciprocal_radius)
    img.set_data(s.data)

    rotation_matrices = generate_rotation_list(structure, phi, theta, psi, max_theta, resolution)
    update_rotation(rotation_matrices)

    fig.canvas.draw_idle()


def update_structure(_ = None):
    global current_structure
    current_structure = (current_structure + 1) % len(structures)

    structure_info = structures[current_structure]
    structure = structure_info['structure']
    dir_a = direction_to_cartesian(structure, (1, 0, 0))
    dir_b = direction_to_cartesian(structure, (0, 1, 0))
    dir_c = direction_to_cartesian(structure, (0, 0, 1))
    dir_a /= np.linalg.norm(dir_a)
    dir_b /= np.linalg.norm(dir_b)
    dir_c /= np.linalg.norm(dir_c)

    ax_a.set_xdata([0, dir_a[0]])
    ax_a.set_ydata([0, dir_a[1]])
    ax_a.set_3d_properties([0, dir_a[2]])
    ax_b.set_xdata([0, dir_b[0]])
    ax_b.set_ydata([0, dir_b[1]])
    ax_b.set_3d_properties([0, dir_b[2]])
    ax_c.set_xdata([0, dir_c[0]])
    ax_c.set_ydata([0, dir_c[1]])
    ax_c.set_3d_properties([0, dir_c[2]])

    btn_zbwz.label.set_text(structure_info['name'])

    update_pattern()


def update_hkl(_ = None):
    h = int(txt_h.text)
    k = int(txt_k.text)
    l = int(txt_l.text)

    structure = structures[current_structure]['structure']
    rotation_angle = angle_between_directions(structure, (0, 0, 1), (h, k, l))
    direction = direction_to_cartesian(structure, (h, k, l))
    direction /= np.linalg.norm(direction)

    axis = np.cross(np.array([0.0, 0.0, 1.0]), direction)
    if np.count_nonzero(axis) == 0:
        # Guard against parallel directions
        axis = np.array([0, 0, 1])
    axis /= np.linalg.norm(axis)

    # The rotation should describe the "camera" rotation, so we need to invert it
    rotation_angle = -rotation_angle

    phi, theta, psi = np.rad2deg(axangle2euler(axis, rotation_angle, axes='rzxz'))
    if phi < 0:   phi   += 360
    if theta < 0: theta += 360
    if psi < 0:   psi   += 360

    slider_phi.set_val(phi)
    slider_theta.set_val(theta)
    slider_psi.set_val(psi)

    update_pattern()


def update_generator(_ = None):
    beam_energy = slider_energy.val
    specimen_thickness = slider_thick.val
    global gen
    gen = pxm.DiffractionGenerator(beam_energy, max_excitation_error=1/specimen_thickness)
    update_pattern()


def hkil_to_hkl(h, k, i, l):
    return (h, k, l)


def uvtw_to_uvw(u, v, t, w):
    U, V, W = 2*u + v, 2*v + u, w
    common_factor = math.gcd(math.gcd(U, V), W)
    return tuple((int(x/common_factor)) for x in (U, V, W))


def plot_lib_2d(library, size, scale, sigma, max_r, phase_name, rotation_list):
    patterns = np.empty((rotation_list.shape[0], rotation_list.shape[1], size, size))

    for i in range(rotation_list.shape[0]):
        for j in range(rotation_list.shape[1]):
            patterns[i, j] = library[phase_name][tuple(np.rad2deg(mat2euler(rotation_list[i, j], 'rzxz')))]['Sim'].as_signal(size, sigma, max_r)

    pxm.ElectronDiffraction(patterns).plot(cmap='viridis_r')


def show_rotation_list(structure_info, reciprocal_angstrom_per_pixel, simulated_gaussian_sigma, beam_energy, specimen_thickness):
    phases = [structure_info['name']]

    # Ångström^{-1}, extent of relrods in reciprocal space. Inverse of specimen thickness is a starting points
    max_excitation_error = 1/specimen_thickness
    reciprocal_radius = reciprocal_angstrom_per_pixel*(half_pattern_size - 1)

    structure = structure_info['structure']
    rotation_list = generate_complete_rotation_list(
        structure,
        *structure_info['corners'],
        [0],
        resolution=np.deg2rad(1))
    # Only one inplane rotation angle, remove this extra dimension
    rotation_list = rotation_list.reshape(rotation_list.shape[0], -1, 3, 3)
    rotation_list_euler = rotation_matrices_to_euler(rotation_list)

    structure_library = StructureLibrary(
            phases,
            [structure],
            [rotation_list_euler])

    gen = pxm.DiffractionGenerator(beam_energy, max_excitation_error=max_excitation_error)
    library_generator = pxm.DiffractionLibraryGenerator(gen)

    diffraction_library = library_generator.get_diffraction_library(
            structure_library,
            calibration=reciprocal_angstrom_per_pixel,
            reciprocal_radius=reciprocal_radius,
            half_shape=(half_pattern_size, half_pattern_size),
            with_direct_beam=False)

    plot_lib_2d(diffraction_library,
            phase_name=structure_info['name'],
            rotation_list=rotation_list,
            size=target_pattern_dimension_pixels,
            scale=reciprocal_angstrom_per_pixel,
            sigma=simulated_gaussian_sigma,
            max_r=reciprocal_radius)

    plt.show()


def update_rotation_list(_ = None):
    reciprocal_angstrom_per_pixel = slider_scale.val
    simulated_gaussian_sigma = slider_sigma.val
    beam_energy = slider_energy.val
    specimen_thickness = slider_thick.val

    structure_info = structures[current_structure]
    update_rotation(generate_complete_rotation_list(
        structure_info['structure'],
        *structure_info['corners'],
        [0],
        resolution=np.deg2rad(2)))

    show_rotation_list(structures[current_structure], reciprocal_angstrom_per_pixel, simulated_gaussian_sigma, beam_energy, specimen_thickness)


gen = pxm.DiffractionGenerator(beam_energy_keV, max_excitation_error=1/specimen_thickness)

fig = plt.figure()

ax_real = fig.add_axes([0.55, 0.25, 0.45, 0.72], projection='3d')
[ax_a], [ax_b], [ax_c], rotation_scatter = plot_3d_axes(ax_real)


ax_img = fig.add_axes([0.05, 0.25, 0.45, 0.72])
img = ax_img.imshow(np.ones((target_pattern_dimension_pixels, target_pattern_dimension_pixels)), vmin=0, vmax=1)
fig.colorbar(img, ax=ax_img)

ax_scale    = plt.axes([0.1, 0.17, 0.4, 0.03])
ax_sigma    = plt.axes([0.1, 0.12, 0.4, 0.03])
ax_energy   = plt.axes([0.1, 0.07, 0.4, 0.03])
ax_thick    = plt.axes([0.1, 0.02, 0.4, 0.03])

ax_phi      = plt.axes([0.6, 0.17, 0.3, 0.03])
ax_theta    = plt.axes([0.6, 0.12, 0.3, 0.03])
ax_psi      = plt.axes([0.6, 0.07, 0.3, 0.03])

ax_h_txt    = plt.axes([0.60, 0.02, 0.04, 0.03])
ax_k_txt    = plt.axes([0.65, 0.02, 0.04, 0.03])
ax_l_txt    = plt.axes([0.70, 0.02, 0.04, 0.03])
ax_hkl_b    = plt.axes([0.75, 0.02, 0.04, 0.03])
ax_zbwz     = plt.axes([0.80, 0.02, 0.04, 0.03])
ax_rot_list = plt.axes([0.85, 0.02, 0.04, 0.03])

slider_scale  = Slider(ax_scale,  'Scale',    0.0, 0.1,  valinit=reciprocal_angstrom_per_pixel, valstep=0.001)
slider_sigma  = Slider(ax_sigma,  '$\\sigma$', 0.0, 0.05, valinit=0.04,  valstep=0.001)
slider_energy = Slider(ax_energy, 'Energy',   100, 300,  valinit=200,   valstep=10)
slider_thick  = Slider(ax_thick,  'Thick',    1,   100,  valinit=80,    valstep=1)

phi_start   = 315
theta_start = 35.264+5
psi_start   = 215
slider_phi    = Slider(ax_phi,    '$\\phi$',    0.0, 360.0, valinit=phi_start,   valstep=0.1)
slider_theta  = Slider(ax_theta,  '$\\theta$', 0.0, 360.0, valinit=theta_start, valstep=0.1)
slider_psi    = Slider(ax_psi,    '$\\psi$',    0.0, 360.0, valinit=psi_start,   valstep=0.1)

txt_h = TextBox(ax_h_txt, 'h', initial='1')
txt_k = TextBox(ax_k_txt, 'k', initial='1')
txt_l = TextBox(ax_l_txt, 'l', initial='2')
btn_hkl = Button(ax_hkl_b, 'Set')
btn_zbwz = Button(ax_zbwz, structures[current_structure]['name'])
btn_rot_list = Button(ax_rot_list, 'List')

slider_scale.on_changed(update_generator)
slider_sigma.on_changed(update_generator)
slider_energy.on_changed(update_generator)
slider_thick.on_changed(update_generator)

slider_phi.on_changed(update_pattern)
slider_theta.on_changed(update_pattern)
slider_psi.on_changed(update_pattern)

btn_hkl.on_clicked(update_hkl)
btn_zbwz.on_clicked(update_structure)
btn_rot_list.on_clicked(update_rotation_list)

update_generator()

plt.show()

