import math

import matplotlib
matplotlib.use('Qt5Agg')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button, TextBox

import diffpy.structure
import pyxem as pxm
from transforms3d.euler import euler2mat
from transforms3d.euler import axangle2euler
from transforms3d.euler import axangle2mat

structure_zb_file = 'D:/Dokumenter/MTNANO/Prosjektoppgave/Data/Gen/NN_test_data/GaAs_mp-2534_conventional_standard.cif'
# structure_zb_file = 'D:/Dokumenter/MTNANO/Prosjektoppgave/Data/Gen/NN_test_data/Al_mp-134_conventional_standard.cif'
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
structure = structure_zb
structure_other = structure_wz
# structure = structure_manual()

gen = pxm.DiffractionGenerator(beam_energy_keV, max_excitation_error=1/specimen_thickness)


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

    rot_count = generate_rotation_list(structure, 0, 0, 0, max_theta, resolution).shape[0]
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
            rotations[i, j] = lattice_to_rotation.T

    return rotations.reshape(-1, 3, 3)


def generate_complete_rotation_list(structure, corner_a, corner_b, corner_c, resolution):
    # Start defining some angles and normals from the given corners
    angle_a_to_b = angle_between_directions(structure, corner_a, corner_b)
    angle_a_to_c = angle_between_directions(structure, corner_a, corner_c)
    angle_b_to_c = angle_between_directions(structure, corner_b, corner_c)
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
    psi_count = math.ceil(angle_b_to_c / resolution)
    rotations = np.zeros((theta_count, psi_count, 3, 3))

    local_cs = []

    # For each theta_count angle theta, evenly spaced
    for i, (theta_b, theta_c) in enumerate(
            zip(np.linspace(0, angle_a_to_b, theta_count),
                np.linspace(0, angle_a_to_c, theta_count))):
        # Define the corner local_b at a rotation theta from corner_a toward
        #   corner_b on the circle surface
        # Similarly, define the corner local_c at a rotation theta from
        #   corner_a toward corner_c

        # The rotation angle is negated, since we rotate the observer instead
        # of the structure itself.
        rotation_a_to_b = axangle2mat(axis_a_to_b, -theta_b)
        rotation_a_to_c = axangle2mat(axis_a_to_c, -theta_c)
        local_b = np.dot(corner_a, rotation_a_to_b)
        local_c = np.dot(corner_a, rotation_a_to_c)
        local_cs.append(local_c)

        # Then define an axis and a maximum rotation to create a great cicle
        # arc between local_b and local_c. Ensure that this is not a degenerate
        # case where local_b and local_c are coincident.
        angle_local_b_to_c = angle_between_directions(structure_zb, local_b, local_c)
        axis_local_b_to_c = np.cross(local_b, local_c)
        if np.count_nonzero(axis_local_b_to_c) == 0:
            # Theta rotation ended at the same position. First position, might
            # be other cases?
            axis_local_b_to_c = corner_a
        axis_local_b_to_c /= np.linalg.norm(axis_local_b_to_c)

        # Generate points along the great circle arc with a distance defined by
        # resolution.
        psi_count_local = math.ceil(angle_local_b_to_c / resolution)
        for j, psi in enumerate(np.linspace(0, angle_local_b_to_c, psi_count_local)):
            # The angle psi is negated as for theta, then the two rotations are
            # combined to give the final rotation matrix.
            rotation_psi = axangle2mat(axis_local_b_to_c, -psi)
            rotations[i, j] = np.matmul(rotation_a_to_b, rotation_psi)
    
    # We also remove duplicates before returning. This eliminates the unused rotations.
    return np.unique(rotations.reshape(-1, 3, 3), axis=0)



def update_rotation(rotation_matrices):
    v = np.empty((rotation_matrices.shape[0], 3))
    for i, rotation_matrix in enumerate(rotation_matrices):
        # v[i] = np.dot(rotation_matrix, np.array([0, 0, 1]))
        v[i] = np.dot(np.array([0, 0, 1]), rotation_matrix)
    rotation_scatter._offsets3d = v.T


def update_pattern(_ = None):
    reciprocal_angstrom_per_pixel = slider_scale.val
    simulated_gaussian_sigma = slider_sigma.val

    phi   = np.deg2rad(slider_phi.val)
    theta = np.deg2rad(slider_theta.val)
    psi   = np.deg2rad(slider_psi.val)

    structure_rotation = euler2mat(phi, theta, psi, axes='rzxz')

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
    global structure
    global structure_other
    structure, structure_other = structure_other, structure
    if structure == structure_zb:
        ax_a.set_xdata([0, 1])
        ax_a.set_ydata([0, 0])
        ax_a.set_3d_properties([0, 0])
        ax_b.set_xdata([0, 0])
        ax_b.set_ydata([0, 1])
        ax_b.set_3d_properties([0, 0])
        ax_c.set_xdata([0, 0])
        ax_c.set_ydata([0, 0])
        ax_c.set_3d_properties([0, 1])
        btn_zbwz.label.set_text('ZB')
    else:
        ax_a.set_xdata([0, 1])
        ax_a.set_ydata([0, 0])
        ax_a.set_3d_properties([0, 0])
        ax_b.set_xdata([0, -0.5])
        ax_b.set_ydata([0, math.sqrt(3)/2])
        ax_b.set_3d_properties([0, 0])
        ax_c.set_xdata([0, 0])
        ax_c.set_ydata([0, 0])
        ax_c.set_3d_properties([0, 1])
        btn_zbwz.label.set_text('WZ')
    update_pattern()


def update_hkl(_ = None):
    h = int(txt_h.text)
    k = int(txt_k.text)
    l = int(txt_l.text)

    rotation_angle = angle_between_directions(structure, (0, 0, 1), (h, k, l))
    if structure == structure_wz:
        # k is along second axis, rotated 30 degrees past the y axis, 120 degrees from the x axis
        # -> [math.cos(np.deg2rad(120)), math.sin(np.deg2rad(120)), 0] = [sqrt(3)/2, -1/2, 0]
        direction_axis = np.array([h, 0, l]) + k*np.array([math.cos(np.deg2rad(120)), math.sin(np.deg2rad(120)), 0])
        # [-0.5, math.sqrt(3)/2, 0])
    else:
        direction_axis = np.array([h, k, l])

    axis = np.cross(np.array([0.0, 0.0, 1.0]), direction_axis)
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


def direction_to_cartesian(structure_from, direction_from):
    # From formula for change of basis, see hand written description
    a = structure_from.lattice.a
    b = structure_from.lattice.b
    c = structure_from.lattice.c
    alpha = np.deg2rad(structure_from.lattice.alpha)  # angle a to c
    beta  = np.deg2rad(structure_from.lattice.beta)    # angle b to c
    gamma = np.deg2rad(structure_from.lattice.gamma)  # angle a to b

    cos_alpha = math.cos(alpha)
    cos_beta = math.cos(beta)
    cos_gamma = math.cos(gamma)
    sin_gamma = math.sin(gamma)

    transform_e1 = np.array([
        a,
        0,
        0])
    transform_e2 = np.array([
        b*cos_gamma,
        b*sin_gamma,
        0])

    factor_e3_0 = cos_beta
    factor_e3_1 = (cos_alpha - cos_beta*cos_gamma)/sin_gamma
    assert(np.dot(factor_e3_0, factor_e3_0) + np.dot(factor_e3_1, factor_e3_1) < 1)  # TODO: Temporary?
    factor_e3_2 = math.sqrt(1 - np.dot(factor_e3_0, factor_e3_0) - np.dot(factor_e3_1, factor_e3_1))
    transform_e3 = np.array([
        c*factor_e3_0,
        c*factor_e3_1,
        c*factor_e3_2])

    transform = np.array([
        transform_e1,
        transform_e2,
        transform_e3]).T

    return np.dot(transform, direction_from)



fig = plt.figure()

ax_real = fig.add_axes([0.55, 0.25, 0.45, 0.72], projection='3d')
[ax_a], [ax_b], [ax_c], rotation_scatter = plot_3d_axes(ax_real)


ax_img = fig.add_axes([0.05, 0.25, 0.45, 0.72])
img = ax_img.imshow(np.ones((target_pattern_dimension_pixels, target_pattern_dimension_pixels)), vmin=0, vmax=1)
fig.colorbar(img, ax=ax_img)

ax_scale  = plt.axes([0.1, 0.17, 0.4, 0.03])
ax_sigma  = plt.axes([0.1, 0.12, 0.4, 0.03])
ax_energy = plt.axes([0.1, 0.07, 0.4, 0.03])
ax_thick  = plt.axes([0.1, 0.02, 0.4, 0.03])

ax_phi    = plt.axes([0.6, 0.17, 0.3, 0.03])
ax_theta  = plt.axes([0.6, 0.12, 0.3, 0.03])
ax_psi    = plt.axes([0.6, 0.07, 0.3, 0.03])

ax_h_txt  = plt.axes([0.6,  0.02, 0.07, 0.03])
ax_k_txt  = plt.axes([0.7,  0.02, 0.07, 0.03])
ax_l_txt  = plt.axes([0.8,  0.02, 0.07, 0.03])
ax_hkl_b  = plt.axes([0.9,  0.02, 0.04, 0.03])
ax_zbwz   = plt.axes([0.95, 0.02, 0.04, 0.03])

slider_scale  = Slider(ax_scale,  'Scale',    0.0, 0.1,  valinit=reciprocal_angstrom_per_pixel, valstep=0.001)
slider_sigma  = Slider(ax_sigma,  '$\sigma$', 0.0, 0.05, valinit=0.04,  valstep=0.001)
slider_energy = Slider(ax_energy, 'Energy',   100, 300,  valinit=200,   valstep=10)
slider_thick  = Slider(ax_thick,  'Thick',    1,   100,  valinit=80,    valstep=1)

phi_start   = 315
theta_start = 35.264+5
psi_start   = 215
slider_phi    = Slider(ax_phi,    '$\phi$',    0.0, 360.0, valinit=phi_start,   valstep=0.1)
slider_theta  = Slider(ax_theta,  '$\\theta$', 0.0, 360.0, valinit=theta_start, valstep=0.1)
slider_psi    = Slider(ax_psi,    '$\psi$',    0.0, 360.0, valinit=psi_start,   valstep=0.1)

txt_h = TextBox(ax_h_txt, 'h', initial='1')
txt_k = TextBox(ax_k_txt, 'k', initial='1')
txt_l = TextBox(ax_l_txt, 'l', initial='2')
btn_hkl = Button(ax_hkl_b, 'Set')
btn_zbwz = Button(ax_zbwz, 'ZB')

slider_scale.on_changed(update_generator)
slider_sigma.on_changed(update_generator)
slider_energy.on_changed(update_generator)
slider_thick.on_changed(update_generator)

slider_phi.on_changed(update_pattern)
slider_theta.on_changed(update_pattern)
slider_psi.on_changed(update_pattern)

btn_hkl.on_clicked(update_hkl)
btn_zbwz.on_clicked(update_structure)

update_generator()

update_rotation(generate_complete_rotation_list(
    structure_zb,
    (0, 0, 1),
    (1, 0, 1),
    (1, 1, 1),
    resolution=np.deg2rad(2)))
# update_structure()
# update_rotation(generate_complete_rotation_list(structure_wz,
    # direction_to_cartesian(structure_wz, uvtw_to_uvw(0, 0, 0, 1)),
    # direction_to_cartesian(structure_wz, uvtw_to_uvw(1, 1, -2, 0)),
    # direction_to_cartesian(structure_wz, uvtw_to_uvw(1, 0, -1, 0)),
    # resolution=np.deg2rad(2)))

plt.show()

