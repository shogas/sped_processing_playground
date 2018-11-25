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
from pyxem.generators.indexation_generator import IndexationGenerator

import diffpy.structure

#
# Temporary while waiting for https://github.com/pyxem/pyxem/pull/320
#
if False:
    from pyxem.generators.structure_library_generator import StructureLibraryGenerator
else:
    from pyxem.libraries.structure_library import StructureLibrary


    def uvtw_to_uvw(uvtw):
        """Convert 4-index direction to a 3-index direction.

        Parameters
        ----------
        uvtw : array-like with 4 floats

        Returns
        -------
        uvw : tuple of 4 floats
        """
        u, v, t, w = uvtw
        u, v, w = 2 * u + v, 2 * v + u, w
        common_factor = math.gcd(math.gcd(u, v), w)
        return tuple((int(x / common_factor)) for x in (u, v, w))


    def angle_between_cartesian(a, b):
        return math.acos(max(-1, min(1.0, np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)))))


    def rotation_list_stereographic(structure, corner_a, corner_b, corner_c, inplane_rotations, resolution):
        """Generate a rotation list covering the inverse pole figure specified by three
            corners in cartesian coordinates.

        Parameters
        ----------
        structure : diffpy.structure.Structure
            Structure for which to calculate the rotation list
        corner_a, corner_b, corner_c : tuple
            The three corners of the inverse pole figure, each given by three
            coordinates. The coordinate system is given by the structure lattice.
        resolution : float
            Angular resolution in radians of the generated rotation list.
        inplane_rotations : list
            List of angles in degrees for in-plane rotation of the diffraction
            pattern. This corresponds to the third Euler angle rotation. The
            rotation list will be generated for each of these angles, and combined.
            This should be done automatically, but by including all possible
            rotations in the rotation list, it becomes too large.

            To cover all inplane rotations, use e.g. np.linspace(0, 2*np.pi, 360)

        Returns
        -------
        rotation_list : numpy.array
            Rotations covering the inverse pole figure given as a of Euler
                angles in degress. This `np.array` can be passed directly to pyxem.
        """
        # Convert the crystal directions to cartesian vectors and normalize
        if len(corner_a) == 4:
            corner_a = uvtw_to_uvw(corner_a)
        if len(corner_b) == 4:
            corner_b = uvtw_to_uvw(corner_b)
        if len(corner_c) == 4:
            corner_c = uvtw_to_uvw(corner_c)

        lattice = structure.lattice

        corner_a = np.dot(corner_a, lattice.stdbase)
        corner_b = np.dot(corner_b, lattice.stdbase)
        corner_c = np.dot(corner_c, lattice.stdbase)

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
            raise ValueError('Directions a and b are parallel')
        if np.count_nonzero(axis_a_to_c) == 0:
            raise ValueError('Directions a and c are parallel')

        rotations = []

        # Generate a list of theta_count evenly spaced angles theta_b in the range
        # [0, angle_a_to_b] and an equally long list of evenly spaced angles
        # theta_c in the range[0, angle_a_to_c].
        # Ensure that we keep the resolution also along the direction to the corner
        # b or c farthest away from a.
        theta_count = math.ceil(max(angle_a_to_b, angle_a_to_c) / resolution)
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
            phi_count_local = max(math.ceil(angle_local_b_to_c / resolution), 1)
            for j, phi in enumerate(
                    np.linspace(0, angle_local_b_to_c, phi_count_local)):
                rotation_phi = axangle2mat(axis_local_b_to_c, phi)

                for k, psi in enumerate(inplane_rotations):
                    # Combine the rotations. Order is important. The matrix is
                    # applied from the left, and we rotate by theta first toward
                    # local_b, then across the triangle toward local_c
                    rotation = list(mat2euler(rotation_phi @ rotation_a_to_b, 'rzxz'))
                    rotations.append(np.rad2deg([rotation[0], rotation[1], psi]))

        return np.unique(rotations, axis=0)


    # Inverse pole figure corners for crystal systems
    stereographic_corners = {
        'cubic': [(0, 0, 1), (1, 0, 1), (1, 1, 1)],
        'hexagonal': [(0, 0, 0, 1), (1, 0, -1, 0), (1, 1, -2, 0)],
        'orthorombic': [(0, 0, 1), (1, 0, 0), (0, 1, 0)],
        'tetragonal': [(0, 0, 1), (1, 0, 0), (1, 1, 0)],
        'trigonal': [(0, 0, 0, 1), (0, -1, 1, 0), (1, -1, 0, 0)],
        'monoclinic': [(0, 0, 1), (0, 1, 0), (0, -1, 0)]
    }


    class StructureLibraryGenerator:
        """Generates a structure library for the given phases

        Parameters
        ----------
        phases : list
            Array of three-component phase descriptions, where the phase
            description is [<phase name> : string, <structure> :
            diffpy.structure.Structure, <crystal system> : string], and crystal
            system is one of 'cubic', 'hexagonal', 'orthorombic', 'tetragonal',
            'trigonal', 'monoclinic'.

        Attributes
        ----------
        phase_names : list of string
            List of phase names.
        structures : list of diffpy.structure.Structure
            List of structures.
        systems : list of string
            List of crystal systems.

        Examples
        --------
        >>> gen = StructureLibraryGenerator([
        ...     ('ZB', structure_zb, 'cubic'),
        ...     ('WZ', structure_wz, 'hexagonal')])
        """

        def __init__(self, phases):
            self.phase_names = [phase[0] for phase in phases]
            self.structures = [phase[1] for phase in phases]
            self.systems = [phase[2] for phase in phases]

        def get_orientations_from_list(self, orientations):
            """Create a structure library from a list of rotations.

            Parameters
            ----------
            orientations : list
                A list over identifiers of lists of euler angles (as tuples) in the rzxz
                convention and in degrees.

            Returns
            -------
            structure_library : StructureLibrary
                Structure library for the given phase names, structures and orientations.
            """
            return StructureLibrary(self.phase_names, self.structures, orientations)

        def get_orientations_from_stereographic_triangle(self, inplane_rotations, resolution):
            """
            Create a structure library from the stereographic triangles of the
            given crystal systems.

            Parameters
            ----------
            inplane_rotations : list
                List over identifiers of lists of inplane rotations of the
                diffraction patterns, in degrees.
            resolution : float
                Rotation list resolution in degrees.

            Returns
            -------
            structure_library : StructureLibrary
                Structure library for the given phase names, structures and crystal system.
            """
            rotation_lists = [
                rotation_list_stereographic(structure, *stereographic_corners[system], inplane_rotation, resolution)
                for phase_name, structure, system, inplane_rotation in
                zip(self.phase_names, self.structures, self.systems, inplane_rotations)]
            return StructureLibrary(self.phase_names, self.structures, rotation_lists)














# Global constants
beam_energy_keV = 200
specimen_thickness = 80  # Ångström
target_pattern_dimension_pixels = 144
half_pattern_size = target_pattern_dimension_pixels // 2
simulated_gaussian_sigma = 0.04
reciprocal_angstrom_per_pixel = 0.032 # From 110 direction, compared to a_crop


# Structure definitions

def structure_manual():
    # This seems to give the same result as structure_zb_file
    a = 5.75
    lattice = diffpy.structure.lattice.Lattice(a, a, a, 90, 90, 90)
    atom_list = []
    for x, y, z in [(0, 0, 0), (0.5, 0, 0.5), (0, 0.5, 0.5), (0.5, 0.5, 0)]:
        atom_list.append(diffpy.structure.atom.Atom(atype='Ga', xyz=[x,      y,      z],      lattice=lattice))
        atom_list.append(diffpy.structure.atom.Atom(atype='As', xyz=[x+0.25, y+0.25, z+0.25], lattice=lattice))
    return diffpy.structure.Structure(atoms=atom_list, lattice=lattice)


structure_zb_file = 'D:\\Dokumenter/MTNANO/Prosjektoppgave/Data/Gen/NN_test_data/GaAs_mp-2534_conventional_standard.cif'
structure_wz_file = 'D:\\Dokumenter/MTNANO/Prosjektoppgave/Data/Gen/NN_test_data/GaAs_mp-8883_conventional_standard.cif'
structure_orthorombic_file = 'D:\\Dokumenter/MTNANO/Prosjektoppgave/Data/Fe2Al5_SM_1201135.cif'
structure_monoclinic_file = 'D:\\Dokumenter/MTNANO/Prosjektoppgave/Data/FeAl3_SM_sd_0261951.cif'

structure_zb = diffpy.structure.loadStructure(structure_zb_file)
structure_wz = diffpy.structure.loadStructure(structure_wz_file)
structure_orthorombic = diffpy.structure.loadStructure(structure_orthorombic_file)
structure_monoclinic = diffpy.structure.loadStructure(structure_monoclinic_file)

# TODO: Actual struture files
structure_tetragonal_file = 'D:\\Dokumenter/MTNANO/Prosjektoppgave/Data/Gen/NN_test_data/GaAs_mp-2534_conventional_standard.cif'
structure_tetragonal = diffpy.structure.loadStructure(structure_tetragonal_file)
structure_tetragonal.lattice.a = 1
structure_tetragonal.lattice.b = 1
structure_tetragonal.lattice.c = 2

structure_trigonal_file = 'D:\\Dokumenter/MTNANO/Prosjektoppgave/Data/Gen/NN_test_data/GaAs_mp-2534_conventional_standard.cif'
structure_trigonal = diffpy.structure.loadStructure(structure_trigonal_file)
structure_trigonal.lattice.a = 1
structure_trigonal.lattice.b = 1
structure_trigonal.lattice.c = 1
structure_trigonal.lattice.alpha = 100
structure_trigonal.lattice.beta = 100
structure_trigonal.lattice.gamma = 100

structures = [
    {
        'name': 'ZB',
        'structure': structure_zb,
        'system': 'cubic',
    },
    {
        'name': 'WZ',
        'structure': structure_wz,
        'system': 'hexagonal',
    },
    {
        'name': 'ort',
        'structure': structure_orthorombic,
        'system': 'orthorombic',
    },
    {
        'name': 'tet',
        'structure': structure_tetragonal,
        'system': 'tetragonal',
    },
    {
        'name': 'tri',
        'structure': structure_trigonal,
        'system': 'trigonal',
    },
    {
        'name': 'mon',
        'structure': structure_monoclinic,
        'system': 'monoclinic',
    }
]
current_structure = 0
current_rotation_list = None


# TODO: Look at https://www.ctcms.nist.gov/%7Elanger/oof2man/RegisteredClass-Bunge.html
def generate_rotation_list_directed(structure, phi, theta, psi, max_theta, resolution):
    # TODO: Symmetry considerations

    zone_to_rotation = np.identity(3)
    lattice_to_zone = np.identity(3)
    lattice_to_zone = euler2mat(phi, -theta, psi, 'rzxz')

    # This generates rotations around the given axis, with a denser sampling close to the axis
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


def generate_fibonacci_spiral(structure, phi, theta, psi, max_theta, _):
    # Vogel's method -> disk. Modify -> sphere surface
    zone_to_rotation = np.identity(3)
    lattice_to_zone = np.identity(3)

    lattice_to_zone = euler2mat(phi, theta, psi, 'rzxz')

    n = 100
    golden_angle = np.pi * (3 - np.sqrt(5))
    theta = golden_angle * np.arange(n)
    z = np.linspace(1, np.cos(max_theta), n)

    radius = np.sqrt(1 - z**2)

    points = np.zeros((n, 3))
    points[:, 0] = radius * np.cos(theta)
    points[:, 1] = radius * np.sin(theta)
    points[:, 2] = z

    rotations = np.empty((n, 3, 3))
    for i, point in enumerate(points):
        # Simplifications to cos angle formula since one of the directions is (0, 0, 1)
        point_angle = math.acos(point[2]/np.linalg.norm(point))
        point_axis = np.cross(np.array([0, 0, 1]), point)
        if np.count_nonzero(point_axis) == 0:
            point_axis = np.array([0, 0, 1])
        zone_to_rotation = axangle2mat(point_axis, point_angle)
        rotations[i] = lattice_to_zone @ zone_to_rotation

    return rotations.reshape(-1, 3, 3)


def plot_3d_axes(ax):
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    axis_a = ax.plot([0, 1], [0, 0], [0, 0], c=(1, 0, 0))
    axis_b = ax.plot([0, 0], [0, 1], [0, 0], c=(0, 1, 0))
    axis_c = ax.plot([0, 0], [0, 0], [0, 1], c=(0, 0, 1))

    rot_count = 1  # Scatter plot seems to add new points automatically, just create one point for a start
    scatter_collection = ax.scatter([0]*rot_count, [0]*rot_count, [0]*rot_count,
            picker=True, pickradius=3.0, cmap='viridis_r', depthshade=False)

    return axis_a, axis_b, axis_c, scatter_collection


def transformation_matrix_to_cartesian(structure):
    """Create the transformation matrix for a change of basis from the
    coordinate system given by the structure lattice to cartesian.

    The function implements a change of basis from the coordinate system specified
    by the structure lattice to a cartesian coordinate system with a standard basis
    [1, 0, 0], [0, 1, 0], [0, 0, 1]. The transformation matrix is greatly simplified
    from the general case since the target basis set is cartesian.

    See https://en.wikipedia.org/wiki/Change_of_basis or any text covering introductory
    linear algebra.

    Parameters
    ----------
        structure : diffpy.structure.Structure
            Structure with a lattice defining the coordinate system
        direction : array-like of three floats
            3D vector to be converted

    Returns
    -------
        transformed_direction : np.array
            Direction with basis changed to cartesian.
    """
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


def crystal_dot(metric_tensor, a, b):
    """Dot product between directions in arbitrary crystal system.

    Parameters
    ----------
    metric_tensor : np.array
        Metric tensor for the crystal system.
    a, b : array-like
        The two direction to compute the dot product between.

    Returns
    -------
    dot_product : float
        Dot product between a and b in the coordinate system specified by
        metric_tensor.
    """
    return np.dot(a, np.dot(metric_tensor, b))


def angle_between_directions(structure, direction_1, direction_2):
    """Angle between directions in the coordinate system given by the structure
    lattice.

    Parameters
    ----------
    structure : diffpy.structure.Structure
        Structure in which to compute the angle.
    direction_1, direction_2 : array-like
        Two directions specified in the coordinate system given by the
        structure lattice.

    Returns
    -------
    angle : float
        Angle between direction_1 and direction_2 in radians.
    """
    metrics = structure.lattice.metrics
    len_1_squared = crystal_dot(metrics, direction_1, direction_1)
    len_2_squared = crystal_dot(metrics, direction_2, direction_2)
    return math.acos(crystal_dot(metrics, direction_1, direction_2) / math.sqrt(len_1_squared * len_2_squared))


def equispaced_s2_grid(theta_range, phi_range, resolution=0.05, no_center=False):
    """Creates rotations approximately equispaced on a sphere.
    Parameters
    ----------
    theta_range : tuple of float
        (theta_min, theta_max)
        The range of allowable polar angles.
    phi_range : tuple of float
        (phi_min, phi_max)
        The range of allowable azimuthal angles.
    resolution : float
        The angular resolution of the grid.
    no_center : bool
        If true, `theta` values will not start at zero.
    Returns
    -------
    s2_grid : array-like
        Each row contains `(theta, phi)`, the azimthal and polar angle
        respectively.
    """
    from decimal import Decimal, ROUND_HALF_UP
    theta_min, theta_max = [t for t in theta_range]
    phi_min, phi_max = [r for r in phi_range]
    resolution = resolution
    resolution = 2 * theta_max / int(Decimal(2 * theta_max / resolution).quantize(0, ROUND_HALF_UP))
    n_theta = int(Decimal((2 * theta_max / resolution + no_center)).quantize(0, ROUND_HALF_UP) / 2)

    if no_center:
        theta_grid = np.arange(0.5, n_theta + 0.5) * resolution
    else:
        theta_grid = np.arange(n_theta + 1) * resolution

    phi_grid = []
    for j, theta in enumerate(theta_grid):
        steps = max(round(math.sin(theta) * phi_max / theta_max * n_theta), 1)
        phi = phi_min\
            + np.arange(steps) * (phi_max - phi_min) / steps \
            + ((j+1) % 2) * (phi_max - phi_min) / steps / 2
        phi_grid.append(phi)
    s2_grid = np.array(
        [(theta, phi, 0) for phis, theta in zip(phi_grid, theta_grid) for phi in
         phis])
    return s2_grid


def equispaced_so3_grid(alpha_max, beta_max, gamma_max, resolution=2.5,
                        alpha_min=0, beta_min=0, gamma_min=0):
    """Creates an approximately equispaced SO(3) grid.
    Parameters
    ----------
    alpha_max : float
    beta_max : float
    gamma_max : float
    resolution : float, optional
    alpha_min : float, optional
    beta_min : float, optional
    gamma_min : float, optional
    Returns
    -------
    so3_grid : array-like
        Each row contains `(alpha, beta, gamma)`, the three Euler angles on the
        SO(3) grid.
    """

    def no_center(res):
        if round(2 * np.pi / res) % 2 == 0:
            return True
        else:
            return False

    s2_grid = equispaced_s2_grid(
        (beta_min, beta_max),
        (alpha_min, alpha_max),
        resolution,
        no_center=no_center(resolution)
    )

    gamma_max = gamma_max / 2

    ap2 = int(np.round(2 * gamma_max / resolution))
    beta, alpha = s2_grid[:, 0], s2_grid[:, 1]
    real_part = np.cos(beta) * np.cos(alpha) + np.cos(alpha)
    imaginary_part = -(np.cos(beta) + 1) * np.sin(alpha)
    d_gamma = np.arctan2(imaginary_part, real_part)
    d_gamma = np.tile(d_gamma, (ap2, 1))
    gamma = -gamma_max + np.arange(ap2) * 2 * gamma_max / ap2
    gamma = (d_gamma + np.tile(gamma.T, (len(s2_grid), 1)).T).flatten()
    alpha = np.tile(alpha, (ap2, 1)).flatten()
    beta = np.tile(beta, (ap2, 1)).flatten()
    so3_grid = np.vstack((alpha, beta, gamma)).T
    return so3_grid


def rotation_matrices_to_euler(rotation_matrices):
    """Convert a rotation list in matrix form to Euler angles in degrees.

    Parameters
    ----------
    rotation_matrices: np.array
        Three or more dimensions, where the last two correspond the 3x3 matrix

    Returns
    -------
        Rotation list in Euler angles in degrees with duplicates removed.
    """
    # Remove duplicates
    rotation_matrices = np.unique(rotation_matrices.reshape(-1, 3, 3), axis=0)
    # Convert to euler angles in degrees
    return np.rad2deg([mat2euler(rotation_matrix, 'rzxz') for rotation_matrix in rotation_matrices])


def rotation_euler_to_matrices(rotation_list):
    """Convert a list of Euler angles in degrees (rzxz) to rotation matrices.

    Parameters
    ----------
    rotation_list: np.array
        List of rotations in Euler angles in degrees (rzxz convention)

    Returns
    -------
        List of rotation matrices of shape [len(rotation_list), 3, 3]
    """
    # Convert to euler angles in degrees
    return np.array([euler2mat(*np.deg2rad(r), 'rzxz') for r in rotation_list])


def update_rotation(rotation_list, colors):
    rotation_matrices = rotation_euler_to_matrices(rotation_list)
    v = np.empty((rotation_matrices.shape[0], 3))
    for i, rotation_matrix in enumerate(rotation_matrices):
        v[i] = np.dot(rotation_matrix, np.array([0, 0, 1]).T)
    if colors is not None:
        colors = np.array(colors)
        colors -= colors.min()
        colors /= colors.max()
        colors = colors**2
        rotation_scatter.set_array(np.array(colors))
        rotation_scatter.set_cmap('viridis_r')
        rotation_scatter.update_scalarmappable()
        rotation_scatter._facecolor3d = rotation_scatter.get_facecolor()
        rotation_scatter._edgecolor3d = rotation_scatter.get_facecolor()
        rotation_scatter.stale = True

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

    max_theta = np.deg2rad(5)
    resolution = np.deg2rad(10)
    rotation_matrices = generate_rotation_list(structure, phi, theta, psi, max_theta, resolution)
    update_rotation(rotation_matrices_to_euler(rotation_matrices), None)

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


def update_uvw(_ = None):
    u = int(txt_u.text)
    v = int(txt_v.text)
    w = int(txt_w.text)

    structure = structures[current_structure]['structure']
    direction = direction_to_cartesian(structure, (u, v, w))
    direction /= np.linalg.norm(direction)
    rotation_angle = angle_between_cartesian((0, 0, 1), direction)
    # rotation_angle = np.deg2rad(lattice.angle((0, 0, 1), (u, v, w)))

    axis = np.cross(np.array([0.0, 0.0, 1.0]), direction)
    if np.count_nonzero(axis) == 0:
        # Guard against parallel directions
        axis = np.array([0, 0, 1])
    axis /= np.linalg.norm(axis)

    phi, theta, psi = np.rad2deg(axangle2euler(axis, rotation_angle, axes='rzxz'))
    if phi < 0:   phi   += 360
    if theta < 0: theta += 360
    if psi < 0:   psi   += 360

    slider_phi.eventson = False  # Prevent set_val from running update_pattern multiple times
    slider_phi.set_val(phi)
    slider_theta.set_val(theta)
    slider_psi.set_val(psi)
    slider_phi.eventson = True

    global current_rotation_list
    current_rotation_list = None
    update_pattern()


def update_generator(_ = None):
    beam_energy = slider_energy.val
    specimen_thickness = slider_thick.val
    global gen
    gen = pxm.DiffractionGenerator(beam_energy, max_excitation_error=1/specimen_thickness)
    update_pattern()


def hkil_to_hkl(h, k, i, l):
    return (h, k, l)


def update_rotation_list(_ = None):
    reciprocal_angstrom_per_pixel = slider_scale.val
    simulated_gaussian_sigma = slider_sigma.val
    beam_energy = slider_energy.val
    specimen_thickness = slider_thick.val

    structure_info = structures[current_structure]
    phase_name = structure_info['name']
    structure = structure_info['structure']

    # Ångström^{-1}, extent of relrods in reciprocal space. Inverse of specimen thickness is a starting point
    max_excitation_error = 1/specimen_thickness
    reciprocal_radius = reciprocal_angstrom_per_pixel*(half_pattern_size - 1)
    resolution = np.deg2rad(1)

    structure_library_generator = StructureLibraryGenerator([
        (phase_name, structure, structure_info['system'])])
    structure_library = structure_library_generator.get_orientations_from_stereographic_triangle([[0]], resolution)

    gen = pxm.DiffractionGenerator(beam_energy, max_excitation_error=max_excitation_error)
    library_generator = pxm.DiffractionLibraryGenerator(gen)

    diffraction_library = library_generator.get_diffraction_library(
            structure_library,
            calibration=reciprocal_angstrom_per_pixel,
            reciprocal_radius=reciprocal_radius,
            half_shape=(half_pattern_size, half_pattern_size),
            with_direct_beam=False)

    global current_rotation_list
    current_rotation_list = structure_library.orientations[0]

    global current_rotation_list_signals
    current_rotation_list_signals = []
    rotation_list_colors = []
    for rotation in current_rotation_list:
        signal = diffraction_library.get_library_entry(phase_name, tuple(rotation))['Sim'].as_signal(
                target_pattern_dimension_pixels,
                simulated_gaussian_sigma,
                reciprocal_radius)
        current_rotation_list_signals.append(signal)
        rotation_list_colors.append(np.sum(signal.data))

    update_rotation(current_rotation_list, rotation_list_colors)
    fig.canvas.draw_idle()


def update_scatter_pick(event):
    if current_rotation_list is not None:
        scatter_point_index = event.ind[0]
        img.set_data(current_rotation_list_signals[scatter_point_index])


gen = pxm.DiffractionGenerator(beam_energy_keV, max_excitation_error=1/specimen_thickness)

# Choose local rotation list method
generate_rotation_list = generate_fibonacci_spiral

fig = plt.figure('Pattern visualizer')

ax_real = fig.add_axes([0.55, 0.25, 0.45, 0.72], projection='3d')
[ax_a], [ax_b], [ax_c], rotation_scatter = plot_3d_axes(ax_real)
fig.canvas.mpl_connect('pick_event', update_scatter_pick)


ax_img = fig.add_axes([0.05, 0.25, 0.45, 0.72])
img = ax_img.imshow(
    np.ones((target_pattern_dimension_pixels, target_pattern_dimension_pixels)),
    vmin=0, vmax=1,
    cmap='gray'
    )
fig.colorbar(img, ax=ax_img)

ax_scale    = plt.axes([0.1, 0.17, 0.4, 0.03])
ax_sigma    = plt.axes([0.1, 0.12, 0.4, 0.03])
ax_energy   = plt.axes([0.1, 0.07, 0.4, 0.03])
ax_thick    = plt.axes([0.1, 0.02, 0.4, 0.03])

ax_phi      = plt.axes([0.6, 0.17, 0.3, 0.03])
ax_theta    = plt.axes([0.6, 0.12, 0.3, 0.03])
ax_psi      = plt.axes([0.6, 0.07, 0.3, 0.03])

ax_u_txt    = plt.axes([0.60, 0.02, 0.04, 0.03])
ax_v_txt    = plt.axes([0.65, 0.02, 0.04, 0.03])
ax_w_txt    = plt.axes([0.70, 0.02, 0.04, 0.03])
ax_uvw_b    = plt.axes([0.75, 0.02, 0.04, 0.03])
ax_zbwz     = plt.axes([0.80, 0.02, 0.04, 0.03])
ax_rot_list = plt.axes([0.85, 0.02, 0.04, 0.03])

slider_scale  = Slider(ax_scale,  'Scale',    0.0, 0.1,  valinit=reciprocal_angstrom_per_pixel, valstep=0.001, valfmt="%1.3f")
slider_sigma  = Slider(ax_sigma,  '$\\sigma$', 0.0, 0.05, valinit=simulated_gaussian_sigma,  valstep=0.001, valfmt="%1.3f")
slider_energy = Slider(ax_energy, 'Energy',   100, 300,  valinit=beam_energy_keV,   valstep=10, valfmt="%1.0f")
slider_thick  = Slider(ax_thick,  'Thick',    1,   100,  valinit=specimen_thickness,    valstep=1, valfmt="%1.0f")

phi_start   = 0.
theta_start = 0.
psi_start   = 0.
slider_phi    = Slider(ax_phi,    '$\\phi$',    0.0, 360.0, valinit=phi_start,   valstep=0.1)
slider_theta  = Slider(ax_theta,  '$\\theta$', 0.0, 360.0, valinit=theta_start, valstep=0.1)
slider_psi    = Slider(ax_psi,    '$\\psi$',    0.0, 360.0, valinit=psi_start,   valstep=0.1)

txt_u = TextBox(ax_u_txt, 'u', initial='0')
txt_v = TextBox(ax_v_txt, 'v', initial='0')
txt_w = TextBox(ax_w_txt, 'w', initial='1')
btn_uvw = Button(ax_uvw_b, 'Set')
btn_zbwz = Button(ax_zbwz, structures[current_structure]['name'])
btn_rot_list = Button(ax_rot_list, 'List')

slider_scale.on_changed(update_generator)
slider_sigma.on_changed(update_generator)
slider_energy.on_changed(update_generator)
slider_thick.on_changed(update_generator)

slider_phi.on_changed(update_pattern)
slider_theta.on_changed(update_pattern)
slider_psi.on_changed(update_pattern)

btn_uvw.on_clicked(update_uvw)
btn_zbwz.on_clicked(update_structure)
btn_rot_list.on_clicked(update_rotation_list)

update_generator()

plt.show()

