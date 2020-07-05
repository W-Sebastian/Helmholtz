from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
from typing import List
from scipy.linalg import qr, eigh


# Implementation of matlab A\B operator (mldivide)
def solve_mldivide(A, b):
    x1, res, rnk, s = np.linalg.lstsq(A, b, rcond=None)
    if rnk == A.shape[1]:
        return x1  # nothing more to do if A is full-rank
    Q, R, P = qr(A.T, mode='full', pivoting=True)
    Z = Q[:, rnk:].conj()
    C = np.linalg.solve(Z[rnk:], -x1[rnk:])
    return x1 + Z.dot(C)


def lobatto_shape_function(xi, order) -> (List[float], List[float]):
    sf = [
        (1 - xi) / 2,
        (1 + xi) / 2,
        (1 / 2) * (3 / 2) ** (1 / 2) * (-1 + xi ** 2),
        (1 / 2) * (5 / 2) ** (1 / 2) * (xi * ((-1) + xi ** 2)),
        (1 / 8) * (7 / 2) ** (1 / 2) * (1 + xi ** 2 * ((-6) + 5 * xi ** 2)),
        (3 / 8) * 2 ** (-1 / 2) * (xi * (3 + xi ** 2 * ((-10) + 7 * xi ** 2))),
        (1 / 16) * (11 / 2) ** (1 / 2) * ((-1) + xi ** 2. * (15 + xi ** 2. * ((-35) + 21. * xi ** 2))),
        (1 / 16) * (13 / 2) ** (1 / 2) * (xi * ((-5) + xi ** 2. * (35 + xi ** 2 * ((-63) + 33 * xi ** 2)))),
        (1 / 128) * (15 / 2) ** (1 / 2) * (
                    5 + xi ** 2. * ((-140) + xi ** 2 * (630 + xi ** 2 * ((-924) + 429 * xi ** 2))))
    ]

    dsf = [
        -1. / 2 + xi * 0,
        1. / 2 + xi * 0,
        (1 / 2) * (3 / 2) ** (1 / 2) * (2. * xi),
        (1 / 2) * (5 / 2) ** (1 / 2) * ((-1) + 3 * xi ** 2),
        (1 / 8) * (7 / 2) ** (1 / 2) * (xi * ((-12) + 20. * xi ** 2)),
        (3 / 8) * 2 ** (-1 / 2) * (3 + xi ** 2. * ((-30) + 35. * xi ** 2)),
        (1 / 16) * (11 / 2) ** (1 / 2) * (xi * (30 + xi ** 2. * ((-140) + 126. * xi ** 2))),
        (1 / 16) * (13 / 2) ** (1 / 2) * ((-5) + xi ** 2 * (105 + xi ** 2 * ((-315) + 231 * xi ** 2))),
        (1 / 128) * (15 / 2) ** (1 / 2) * (xi * ((-280) + xi ** 2. * (2520 + xi ** 2. * ((-5544) + 3432. * xi ** 2))))
    ]

    sf_factors = []
    dsf_factors = []

    for i in range(0, order + 1):
        sf_factors.append(sf[i])
        dsf_factors.append(dsf[i])
    return [sf_factors, dsf_factors]


def plot_lobatto_shape_functions():
    x_values = list(np.arange(-1, 1.01, 0.01))
    y_values = []
    dy_values = []

    order = 8

    for i in range(1, order + 2):
        y_values.append([])
        dy_values.append([])

    for i in x_values:
        sf, dsf = lobatto_shape_function(i, order)
        for orderIdx in range(0, order + 1):
            y_values[orderIdx].append(sf[orderIdx])
            dy_values[orderIdx].append(dsf[orderIdx])

    # use the plot function
    plt.style.use('seaborn-darkgrid')

    fig, axs = plt.subplots(4, 2, figsize=(12, 9))

    for rowIdx in range(0, 4):
        for colIdx in range(0, 2):
            max_order = rowIdx * 2 + colIdx + 1

            axs[rowIdx, colIdx].set(xlabel=r'$\xi$', ylabel=r"$l(\xi)$")
            axs[rowIdx, colIdx].set_title('Order {}'.format(max_order))
            for orderIdx in range(0, max_order + 1):
                axs[rowIdx, colIdx].plot(x_values, y_values[orderIdx])

    fig.show()

    fig, axs = plt.subplots(4, 2, figsize=(12, 9))

    for rowIdx in range(0, 4):
        for colIdx in range(0, 2):
            max_order = rowIdx * 2 + colIdx + 1

            axs[rowIdx, colIdx].set(xlabel=r'$\xi$', ylabel=r"$l(\xi)$")
            axs[rowIdx, colIdx].set_title('Order {}'.format(max_order))
            yc = y_values[0]
            for orderIdx in range(1, max_order + 1):
                yc = [a + b for a, b in zip(yc, y_values[orderIdx])]
            axs[rowIdx, colIdx].plot(x_values, dy_values[max_order])

    fig.show()


def print_lobatto_shape_functions():
    for i in range(1, 9):
        print("Index of {}".format(i))
        sf, dsf = lobatto_shape_function(0, i)
        print("{}".format(sf))
        print("{}".format(dsf))


# A coordinate in 1 dimensional space
class Coord1d:
    def __init__(self, x):
        self.x = x

    def __str__(self):
        return "{}".format(self.x)

    def __repr__(self):
        return self.__str__()


# A 1D linear element; the element is based on start and end nodes
# The element supports p-fem polynomial refinement where we also have interior DOFs created
class Element1d:
    def __init__(self, first_node, second_node):
        self.nodes = [first_node, second_node]

        self.p_fem = 1
        # When p_fem is higher than 1 then we also need
        # a start idx for internal DOFs
        self.interior_dofs_start_id = 0

    def set_p_fem(self, order: int):
        self.p_fem = order

    def internal_dof_indices(self):
        return list(range(self.interior_dofs_start_id, self.interior_dofs_start_id + self.p_fem - 1, 1))

    def __str__(self):
        return "[{}, {}]".format(self.nodes[0], self.nodes[1])

    def __repr__(self):
        return self.__str__()

    def num_element_dofs(self):
        return len(self.nodes)

    def num_p_fem_dofs(self):
        return self.p_fem - 1


class Mesh1d:
    def __init__(self):
        self.node_coordinates = []
        self.elements = []

    def num_dofs(self):
        return sum(element.num_p_fem_dofs() for element in self.elements) + len(self.node_coordinates)

    def generate_internal_dof_indices(self):
        last_idx = len(self.node_coordinates)
        for element in self.elements:
            if element.p_fem == 1:
                continue

            element.interior_dofs_start_id = last_idx
            last_idx += (element.p_fem - 1)

    def __str__(self):
        return "{}\n{}".format(self.node_coordinates, self.elements)

    def __repr__(self):
        return self.__str__()


# Creates a mesh for a duct from x=0 to x=L with n elements;
#
# length (L): the length of the duct
# num_elements (n): number of elements to split the domain into
# returns: a 1D mesh contains nodes (1D coordinates) and elements (start/end node)
def create_1d_mesh(length, num_elements):
    num_nodes = num_elements + 1
    coordinates = np.linspace(0, length, num_nodes)

    mesh = Mesh1d()
    mesh.node_coordinates = [Coord1d(x) for x in coordinates]
    for i in range(0, len(coordinates) - 1):
        mesh.elements.append(Element1d(i, i + 1))
    return mesh


# Defines the gauss quadrature; the gauss quadrature contains n-points and n-weights such as
#   sum of (w_i * p_i) will give an approximation of the original function integral
class GaussQuadrature:
    def __init__(self):
        self.points = []
        self.weights = []

    def __init__(self, points, weights):
        self.points = points
        self.weights = weights

    def __str__(self):
        return "{}\n{}".format(self.points, self.weights)


# Compute the gauss legendre quadrature for a 1D element with coordinates [x1, x2]
# This is actually pre-computed in coordinates [-1, 1] and transformed to real coordinates
# For 2D and 3D this is a bit more complicated and ideally we would never do this; it is much better
#   to compute the integral using the gaussian quadrature in ideal coordinates
#   and move the result to the real space (for performance reasons)
# However, this is an academic case in 1D so no such concerns;
def gauss_legendre_quadrature(x1: float, x2: float, function_order: int):
    xm = 0.5 * (x2 + x1)
    xl = 0.5 * (x2 - x1)

    gq = ideal_gauss_legendre_quadrature(function_order)
    # Normally this is done using the shape functions and Jacobian for the correct order and the correct element
    #   because the operation is in fact an interpolation but here we can get away cheap due to 1D being simple
    for idx, point in enumerate(gq.points):
        gq.points[idx] = xm + xl * point
        gq.weights[idx] *= xl

    return gq


# Compute the Gauss legendre quadrature in ideal 1D element for a specified polynomial order
#
# Returns the gauss quadrature for that order
# The results of this functions could potentially be hardcoded for all orders and all element types
# as long as they are always computed in ideal space of [-1, 1] coordinates.
# We don't necessary have a performance concern for 1D cases so we compute always and each time
# There are multiple ways of computing this quadrature but what we actually need to find out is the roots
# of the polynomial of the specified order and their weights;
# see: https://arxiv.org/pdf/1802.03948.pdf for an algorithm of this computation
def ideal_gauss_legendre_quadrature(function_order: int):
    num_integration_points = function_order * 2

    epsilon = 3e-14

    points = np.zeros(num_integration_points)
    weights = np.zeros(num_integration_points)

    for i in range(1, function_order + 1):
        z = np.cos(np.pi * (i - 0.25) / (num_integration_points + 0.5))
        z1 = z + 10

        while np.abs(z - z1) > epsilon:
            p1 = 1
            p2 = 0
            for j in range(1, num_integration_points + 1):
                p3 = p2
                p2 = p1
                p1 = ((2 * j - 1) * z * p2 - (j - 1) * p3) / j
            pp = num_integration_points * (z * p1 - p2) / (z * z - 1)
            z1 = z
            z = z1 - p1 / pp

        points[i - 1] = -z
        points[num_integration_points - (i - 1) - 1] = z

        weights[i - 1] = 2 / ((1 - z ** 2) * pp ** 2)
        weights[num_integration_points - (i - 1) - 1] = weights[i - 1]

    return GaussQuadrature(points, weights)


def write_hardcoded_gauss_quadrature_points():
    with open('gauss_legendre_quadrature.py', 'w') as f:
        f.write('gauss_quadrature = [\n')
        for i in range(1, 9):
            # gq = gauss_legendre_quadrature(i)
            f.write('\tGaussQuadrature(')
        f.write(']\n')


def element_mass_stiffness_matrices(mesh: Mesh1d, element: Element1d):
    p_fem_order = element.p_fem

    Ke = np.zeros((p_fem_order + 1, p_fem_order + 1))
    Me = np.zeros((p_fem_order + 1, p_fem_order + 1))

    x1 = mesh.node_coordinates[element.nodes[0]]
    x2 = mesh.node_coordinates[element.nodes[1]]

    Length = x2.x - x1.x

    gq = ideal_gauss_legendre_quadrature(p_fem_order)

    for idx in range(0, len(gq.points)):
        gauss_point = gq.points[idx]
        gauss_weight = gq.weights[idx]

        xi, dLdxi = lobatto_shape_function(gauss_point, p_fem_order)
        dLdx = [2 / Length * x for x in dLdxi]  # derivative of the shape function in real coordinates

        # Local Stiffness Matrix can be computed from the following equation:
        #   transpose(pressure_field) * integral {transpose(B) * B} dL * pressure_field
        # where the integral is the element stiffness matrix
        # B is a vector of shape functions gradients (derivatives) however we need to get it from local to global space
        # Therefore the integral is the acoustic stiffness matrix but in truth it is not denoting stiffness
        #   but the relation between pressure and acceleration in a finite element

        B = np.array([dLdx])
        # integration at gauss point in real domain using gauss weights
        K = gauss_weight * (np.transpose(B) * B) * Length / 2
        Ke = Ke + K  # we need to sum the stiffness for all gauss points

        # Mass matrix is the same as stiffness matrix but we use a matrix N of shape functions rather than derivatives
        N = np.array([xi])
        M = gauss_weight * (np.transpose(N) * N) * Length / 2
        Me = Me + M

    return Me, Ke


def apply_robin_boundary_condition(Matrix, beta, omega, c0, node_indices):
    indices = np.array(node_indices)
    Matrix[indices[:, None], indices] += 1j * (omega / c0) * beta


def create_velocity_load(velocity, num_system_dofs, load_indices):
    indices = np.array(load_indices)
    rhs = np.zeros((num_system_dofs, 1), dtype=np.complex)
    rhs[indices] += velocity
    return rhs


# Take all element matrices and assemble them in the system matrix
#   Special care must be taken for p-fem DOFs because they don't really exist
def assemble_matrix(mesh: Mesh1d, omega: float, c0: float):
    num_dofs = mesh.num_dofs()

    wavelength = omega / c0
    Matrix = np.zeros((num_dofs, num_dofs), dtype=np.complex)

    for element in mesh.elements:
        # get element mass and stiffness matrix for assembly
        Me, Ke = element_mass_stiffness_matrices(mesh, element)

        # set the element dofs first and then treat the p-fem dofs
        indices = np.array(element.nodes)
        if element.num_p_fem_dofs() > 0:
            indices = np.append(indices, element.internal_dof_indices())

        Matrix[indices[:, None], indices] += Ke - wavelength ** 2 * Me

    return Matrix


def test_matrix_stiffness_mass_one_element():
    mesh = Mesh1d()
    mesh.node_coordinates = [-1, 1]
    mesh.elements = [Element1d(0, 1)]
    mesh.elements[0].p_fem = 4

    M, K = element_mass_stiffness_matrices(mesh, mesh.elements[0])


# Solve Ax = B for x where
#   A is our system matrix
#   B is the rhs matrix
def solve_system_matrix(system_matrix, rhs):
    s = solve_mldivide(system_matrix, rhs)
    return np.array([np.complex(x) for x in s])


def interpolate_in_element_ri(element: Element1d, data: List[np.complex], position, order):
    if order > element.p_fem:
        raise Exception('Requested order higher than element order')

    u1 = data[element.nodes[0]]
    u2 = data[element.nodes[1]]

    sf, _ = lobatto_shape_function(position, element.p_fem)

    v = 0
    v += u1 * sf[0]
    v += u2 * sf[1]

    if order > 1:
        internal_indices = element.internal_dof_indices()
        for idx, dataIdx in enumerate(internal_indices):
            d = data[dataIdx]

            v += d * sf[idx + 2]

    return v


def interpolate_in_element_mp(element: Element1d, data: List[np.complex], position, order):
    if order > element.p_fem:
        raise Exception('Requested order higher than element order')

    u1 = data[element.nodes[0]]
    u2 = data[element.nodes[1]]

    u1_mag = np.abs(u1)
    u1_phase = np.angle(u1)
    u2_mag = np.abs(u2)
    u2_phase = np.angle(u2)

    sf, _ = lobatto_shape_function(position, element.p_fem)

    v_mag = u1_mag * sf[0]
    v_phase = u1_phase * sf[0]

    v_mag += u2_mag * sf[1]
    v_phase += u2_phase * sf[1]

    if order > 1:
        internal_indices = element.internal_dof_indices()
        for idx, dataIdx in enumerate(internal_indices):
            d = data[dataIdx]
            d_mag = np.abs(d)
            d_phase = np.angle(d)

            v_mag += d_mag * sf[idx + 2]
            v_phase += d_phase * sf[idx + 2]

    return v_mag * np.exp(1j*v_phase)


def interpolate_higher_order_solution(mesh: Mesh1d, data: List[np.complex], num_extra_points_per_element):
    values_at_positions = []

    # First add all the values from results that are on the nodes

    for idx, coordinate in enumerate(mesh.node_coordinates):
        values_at_positions.append((coordinate.x, np.complex(data[idx])))

    if num_extra_points_per_element > 0:
        # Generate the interpolation positions in the ideal element
        ideal_positions = np.linspace(-1, 1, num_extra_points_per_element + 2)
        ideal_positions = ideal_positions[1:-1]

        # Now interpolate inside the element and add those values too
        for idx, element in enumerate(mesh.elements):
            x1 = mesh.node_coordinates[element.nodes[0]]
            x2 = mesh.node_coordinates[element.nodes[1]]

            positions = np.linspace(x1.x, x2.x, num_extra_points_per_element + 2)
            # first and last position is for the nodes themselves so we can ignore that
            positions = positions[1:-1]
            values = [
                interpolate_in_element_ri(element, data, xi, element.p_fem)
                for xi in ideal_positions
            ]
            values_at_positions.extend((x, np.complex(v)) for (x, v) in zip(positions, values))

    # sort values based on coordinate
    values_at_positions.sort(key=lambda coord: coord[0])
    x = [a[0] for a in values_at_positions]
    y = [a[1] for a in values_at_positions]
    return np.array(x), np.array(y)


# Models a velocity load applied on a single node at a certain frequency
class VelocityLoad:
    def __init__(self, omega, velocity, node_index):
        self.omega = omega
        self.velocity = velocity
        self.node_index = node_index


# Models an Impedance boundary condition on a certain node
# beta parameters goes from 0 (reflective) to 1 (full admittance)
# beta is defined as acoustic resistance (R) over rho*c (mass density * speed of sound)
class Impedance:
    def __init__(self, beta, node_index):
        self.impedance_values = {node_index: beta}

    def add_impedance(self, beta, node_index):
        self.impedance_values[node_index] = beta


# Defines the fluid properties in terms of mass density and speed of sound
class FluidProperties:
    def __init__(self, rho, c):
        self.rho = rho
        self.c = c


class ComputationParameters:
    def __init__(self, femao_max_order=1):
        self.p_fem = femao_max_order


class HelmholtzSimulation:
    def __init__(self, mesh=None, load=None, impedance=None, fluid_properties=None):
        self.mesh: Mesh1d = mesh
        self.load: VelocityLoad = load
        self.impedance: Impedance = impedance
        self.fluid_properties: FluidProperties = fluid_properties

    def set_mesh(self, mesh: Mesh1d):
        self.mesh = mesh

    def set_load(self, load: VelocityLoad):
        self.load = load

    def set_impedance(self, impedance: Impedance):
        self.impedance = impedance

    def set_fluid_properties(self, fluid_properties: FluidProperties):
        self.fluid_properties = fluid_properties

    def compute(self, computation_settings: ComputationParameters = None):
        if computation_settings is None:
            computation_settings = ComputationParameters()  # initialize with default settings

        # Validate that we have everything we need
        if self.mesh is None:
            raise Exception('Computation requires a valid mesh!')
        if self.load is None:
            raise Exception('Computation requires a load!')
        if self.fluid_properties is None:
            raise Exception('Computation requires fluid properties!')

        # setup FEMAO
        for element in self.mesh.elements:
            element.p_fem = computation_settings.p_fem
        self.mesh.generate_internal_dof_indices()

        # assemble the system matrix
        system_matrix = assemble_matrix(self.mesh, self.load.omega, self.fluid_properties.c)

        # if needed, setup the impedance boundary conditions as a robin BC
        if self.impedance is not None:
            for k, v in self.impedance.impedance_values.items():
                apply_robin_boundary_condition(system_matrix,
                                               v, self.load.omega, self.fluid_properties.c, [k])

        # impose the velocity on the right hand side of the equation
        rhs = create_velocity_load(self.load.velocity, len(system_matrix), [self.load.node_index])

        # solve for the system matrix and RHS
        solution = solve_system_matrix(system_matrix, rhs)

        return solution


def compute_duct_problem():
    plot_ri = True
    plot_mp = False
    plot_sol = False

    # Create the mesh
    num_elements = 24
    length = 2
    mesh = create_1d_mesh(length, num_elements)

    # Setup the fluid properties
    rho0 = 1
    c0 = 1
    fluid_properties = FluidProperties(rho0, c0)

    # Setup load parameters
    omega = 10*np.pi
    Vn = 1 / (rho0 * c0)
    load = VelocityLoad(omega, velocity=1j * omega * rho0 * Vn, node_index=0)

    # Setup impedance condition
    beta = 0.1
    impedance = Impedance(beta, node_index=len(mesh.node_coordinates) - 1)
    # impedance.add_impedance(1, 0)

    # Create the simulation
    sim = HelmholtzSimulation(mesh, load, impedance, fluid_properties)

    for p_fem in [8]:

        solution = sim.compute(ComputationParameters(p_fem))

        x, y = interpolate_higher_order_solution(mesh, solution, 100)

        y_real = [np.real(yi) for yi in y]
        y_imag = [np.imag(yi) for yi in y]
        y_mag = [np.absolute(yi) for yi in y]
        y_phase = [np.angle(yi) for yi in y]

        max_n = len(mesh.node_coordinates)
        xs_real = np.linspace(0, length, max_n)
        ys_real = [np.real(yis) for yis in solution[0:max_n]]
        ys_imag = [np.imag(yis) for yis in solution[0:max_n]]
        ys_mag = [np.abs(yis) for yis in solution[0:max_n]]
        ys_phase = [np.angle(ym) for ym in solution[0:max_n]]

        reference_args = np.linspace(0, length, 100)
        c = (omega*rho0 * Vn) / ((omega/c0)*beta)
        reference_function = np.exp(-1j * omega * reference_args)
        # reference_function = c * np.exp(-1j * omega * reference_args)

        # Real Imaginary plots

        plt.style.use('seaborn-darkgrid')
        if plot_ri:
            fig, axs = plt.subplots(2, 1, figsize=(9, 6))
            plt.ticklabel_format(useOffset=False)
            axs[0].set_title('p-fem={}; {} elements; L={}m; beta={}; omega={:.2f}Hz\n\n'.format(
                p_fem, num_elements, length, beta, omega), fontsize=12)
            axs[0].plot(x, y_real)
            if plot_sol:
                axs[0].plot(xs_real, ys_real,  linestyle='-', marker='o')
            axs[0].plot(reference_args, np.real(reference_function), linestyle=':', marker='+')
            axs[0].set_xlabel('duct length(m)')
            axs[0].set_ylabel('pressure (real)')

            axs[1].plot(x, y_imag)
            if plot_sol:
                axs[1].plot(xs_real, ys_imag,  linestyle='-', marker='o')
            axs[1].plot(reference_args, np.imag(reference_function), linestyle=':', marker='+')
            axs[1].set_xlabel('duct length(m)')
            axs[1].set_ylabel('pressure (imaginary)')
            if plot_sol:
                fig.legend(['Helmholtz', 'Solution', 'Reference'])
            else:
                fig.legend(['Helmholtz', 'Reference'])
            fig.tight_layout()
            fig.show()

        # Magnitude/Phase plots
        if plot_mp:
            fig, axs = plt.subplots(2, 1, figsize=(9, 6))
            plt.ticklabel_format(useOffset=False)
            axs[0].set_title('p-fem={}; {} elements; L={}m; beta={}; omega={:.2f}Hz\n\n'.format(
                p_fem, num_elements, length, beta, omega), fontsize=12)
            axs[0].plot(x, y_mag)
            if plot_sol:
                axs[0].plot(xs_real, ys_mag, linestyle='-', marker='o')
            ref_mag = [np.absolute(i) for i in reference_function]
            axs[0].plot(reference_args, ref_mag, linestyle=':')
            axs[0].set_xlabel('duct length(m)')
            axs[0].set_ylabel('pressure (magnitude)')

            axs[1].plot(x, y_phase)
            if plot_sol:
                axs[1].plot(xs_real, ys_phase, linestyle='-', marker='o')
            axs[1].plot(reference_args, np.angle(reference_function), linestyle=':')
            axs[1].set_xlabel('duct length(m)')
            axs[1].set_ylabel('pressure (phase)')
            if plot_sol:
                fig.legend(['Helmholtz', 'Solution', 'Reference'])
            else:
                fig.legend(['Helmholtz', 'Reference'])
            fig.tight_layout()
            fig.show()

        # M, K = assemble_mass_stiffness_matrices(mesh, omega, c0)
        # eigvals, eigvecs = eigh(K, M, eigvals_only=False)
        #
        # eigvals = [np.sqrt(np.round(e, 6)) for e in eigvals]
        #
        # print(eigvals)
        #
        # fig, axs = plt.subplots(3, 3)
        # mode_offset = 0
        # for i in range(0, 3):
        #     for j in range(0, 3):
        #         idx = mode_offset + i * 3 + j
        #         axs[i, j].plot(x, np.absolute(eigvecs[idx]))
        #         axs[i, j].set_title('{} Hz'.format(eigvals[idx]))
        # fig.show()


def plot_sf_comparison():
    L0 = 0
    L1 = 2

    def target_function(arg):
        return np.sin(np.pi * arg)
        # return arg**2
        # return arg

    x = np.linspace(L0, L1, 100)
    y = target_function(x)

    # plt.plot(x, y)  # reference function
    # legend_labels = ['reference']
    legend_labels = []

    order = 1

    for i in range(2, 11):
        num_elements = i

        coords = np.linspace(L0, L1, num_elements + 1 + (order - 1) * num_elements)
        num_nodes_in_element = order + 1
        values_at_coords = target_function(coords)

        interpolated_coords = []
        interpolated_values = []

        xi = np.linspace(-1, 1, 3)  # interpolate in 20 points

        n1 = ((1 - xi) / 2)
        n2 = ((1 + xi) / 2)

        for eIdx in range(0, num_elements):
            start_idx = eIdx * (num_nodes_in_element - 1)
            locations = coords[start_idx] * n1 + coords[start_idx + num_nodes_in_element - 1] * n2
            values = np.zeros(len(locations))

            sf_indices = [0]
            if order > 1:
                sf_indices.extend(list(range(2, order+1)))
            sf_indices.append(1)

            for xIdx in range(0, len(xi)):
                result = 0
                xi_sf, _ = lobatto_shape_function(xi[xIdx], order)

                # corner nodes
                result += xi_sf[0] * values_at_coords[start_idx]
                result += xi_sf[1] * values_at_coords[start_idx + num_nodes_in_element - 1]

                # internal nodes
                for nIdx in range(0, num_nodes_in_element - 2):
                    result += xi_sf[nIdx + 2] * values_at_coords[start_idx + nIdx + 1]

                values[xIdx] = result
            interpolated_coords.extend(locations)
            interpolated_values.extend(values)

        legend_labels.append('{} elemente'.format(num_elements))
        plt.plot(interpolated_coords, interpolated_values)

    plt.style.use('seaborn-darkgrid')
    # plt.title('Shape functions interpolation for order {}'.format(order))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(legend_labels)
    plt.show()


if __name__ == "__main__":
    # compute_duct_problem()
    plot_sf_comparison()
