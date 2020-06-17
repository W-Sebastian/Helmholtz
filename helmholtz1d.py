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
        self.p_nodes = []
        self.p_fem = 1
        self.p_fem_index = 0

    def set_p_fem(self, order: int):
        self.p_fem = order

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


def create_p_fem_interior_dofs(mesh: Mesh1d, element: Element1d, p_fem: int):
    pass


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
# as long as they are always computed in ideal space of [-1; 1] coordinates.
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
    p_fem_order = element.p_fem  # p-fem order is saved in the element itself

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
    Matrix[indices[:, None], indices] += 1j * omega / c0 * beta


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

    # we keep track of the p-fem rows and columns because we create DOFs which don't really exist
    last_index = len(mesh.node_coordinates)
    for element in mesh.elements:
        # get element mass and stiffness matrix for assembly
        Me, Ke = element_mass_stiffness_matrices(mesh, element)

        # set the element dofs first and then treat the p-fem dofs
        indices = np.array(element.nodes)
        if element.num_p_fem_dofs() > 0:
            indices = np.append(indices, range(last_index, last_index + element.num_p_fem_dofs()))
        last_index += element.num_p_fem_dofs()

        Matrix[indices[:, None], indices] += Ke - wavelength ** 2 * Me

    return Matrix


def assemble_mass_stiffness_matrices(mesh: Mesh1d, omega: float, c0: float):
    num_dofs = mesh.num_dofs()

    K = np.zeros((num_dofs, num_dofs), dtype=np.complex)
    M = np.zeros((num_dofs, num_dofs), dtype=np.complex)

    # we keep track of the p-fem rows and columns because we create DOFs which don't really exist
    last_index = len(mesh.node_coordinates)
    for element in mesh.elements:
        # get element mass and stiffness matrix for assembly
        Me, Ke = element_mass_stiffness_matrices(mesh, element)

        # set the element dofs first and then treat the p-fem dofs
        indices = np.array(element.nodes)
        if element.num_p_fem_dofs() > 0:
            indices = np.append(indices, range(last_index, last_index + element.num_p_fem_dofs()))
        last_index += element.num_p_fem_dofs()

        K[indices[:, None], indices] += Ke
        M[indices[:, None], indices] += (1 / c0 ** 2 * Me)

    return M, K


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
    return solve_mldivide(system_matrix, rhs)


def compute_duct_problem():
    for p_fem in [2]:
        num_elements = 6
        length = 2

        omega = np.pi
        c0 = 1
        rho0 = 1

        beta = 1
        Vn = 1 / (rho0 * c0)

        mesh = create_1d_mesh(length, num_elements)

        for element in mesh.elements:
            element.p_fem = p_fem

        system_matrix = assemble_matrix(mesh, omega, c0)

        # apply robin boundary condition on the last node; this is modeling acoustic impedance
        apply_robin_boundary_condition(system_matrix, beta, omega, c0, [len(mesh.node_coordinates) - 1])

        # apply a velocity load on the first node
        rhs = create_velocity_load(1j * omega * rho0 * Vn, len(system_matrix), [0])

        # solve the system matrix with the created load

        solution = solve_system_matrix(system_matrix, rhs)

        sine_lobatto = [np.imag(x) for x in solution[0:, 0]]
        print(sine_lobatto)

        # Convert the solution to real space with lobatto shape functions
        # interpolate the solution on range between -1 and 1
        num_interpolation_points = 100  # might be too much?
        xi = np.array([-1, 1])

        x = np.array([c.x for c in mesh.node_coordinates])
        y_real = [np.real(x) for x in solution[0:len(mesh.node_coordinates), 0]]
        y_imag = [np.imag(x) for x in solution[0:len(mesh.node_coordinates), 0]]
        y_mag = [np.absolute(x) for x in solution[0:len(mesh.node_coordinates), 0]]
        # do some rounding to get rid of pesky floating point errors
        y_mag = [np.round(x, 1) for x in y_mag]
        reference_function = np.exp(-1j * omega * x)

        plt.style.use('seaborn-darkgrid')
        fig, axs = plt.subplots(3, 1, figsize=(9, 6))

        plt.ticklabel_format(useOffset=False)
        axs[0].plot(x, y_real)
        axs[0].plot(x, np.real(reference_function), linestyle=':', marker='+')
        axs[0].set_xlabel('duct length(m)')
        axs[0].set_ylabel('pressure (real)')

        axs[1].plot(x, y_imag)
        axs[1].plot(x, np.imag(reference_function), linestyle=':', marker='+')
        axs[1].set_xlabel('duct length(m)')
        axs[1].set_ylabel('pressure (imaginary)')

        axs[2].plot(x, y_mag)
        axs[2].plot(x, np.absolute(reference_function), linestyle=':', marker='+')
        axs[2].set_xlabel('duct length(m)')
        axs[2].set_ylabel('pressure (magnitude)')

        fig.legend(['Helmholtz', 'Reference'])
        fig.suptitle('p-fem={} with {} elements and L={}; beta={}'.format(
            p_fem, num_elements, length, beta), fontsize=12)
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
        # return np.sin(np.pi * arg)
        # return arg**2
        return arg

    x = np.linspace(L0, L1, 100)
    y = target_function(x)
    plt.plot(x, y)  # reference function

    legend_labels = ['reference']

    order = 2

    for i in [1]:
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

        legend_labels.append('{} elements'.format(num_elements))
        plt.plot(interpolated_coords, interpolated_values)

    plt.style.use('seaborn-darkgrid')
    plt.title('Shape functions interpolation for order {}'.format(order))
    plt.legend(legend_labels)
    plt.show()


if __name__ == "__main__":
    compute_duct_problem()
    # plot_sf_comparison()
