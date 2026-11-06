"""Bounded batch port of the pinned pymatgen 2024.11.13 reducer.

pymatgen is MIT-licensed; its notice is in THIRD_PARTY_NOTICES.md.
Codes 0/1 certify the Niggli/guard-selected LLL lattice, including signed zeros.
Code 2 asks the caller to use the reference; its matrix is unspecified. Work
limits and numerical boundary checks deliberately narrow the certified domain.
No pymatgen objects, object-mode compilation, or unbounded searches are used.

The operation order follows ``Lattice._calculate_lll``, the A1--A8 Niggli
rules, ``from_parameters``, and ``find_all_mappings``. In particular, the
mapping is against the ORIGINAL basis. Its origin-sphere order is ascending
linked-cell index, then descending image index, as in the pinned Cython
``find_points_in_spheres``. Cartesian mapping vectors use BLAS ``dot``;
replacing that operation with scalar sums changes their last bits. LLL keeps
its successive subtractions instead of reconstructing an integer transform.
"""
from __future__ import annotations

import math

import numpy as np

from orbitol.jit import njit


_LLL_STEPS = 128
_NIGGLI_STEPS = 100
_MAX_IMAGES = 100_000
_MAX_CANDIDATES = 512
_MAX_MATCHES = 100_000
_EPS = np.finfo(np.float64).eps
_BOUNDARY = 128 * _EPS


@njit
def _sum3(a, b):
    """Cython's left-associated three-term sum, without fastmath/FMA."""
    return (a[0] * b[0] + a[1] * b[1]) + a[2] * b[2]


@njit
def _lengths(matrix):
    """Match sqrt(sum(matrix**2, axis=1)), not BLAS vector norms."""
    result = np.empty(3)
    for i in range(3):
        result[i] = math.sqrt(_sum3(matrix[i], matrix[i]))
    return result


@njit
def _near(a, b, scale):
    """Decline comparisons within a conservative rounding envelope."""
    return abs(a - b) <= _BOUNDARY * max(abs(a), abs(b), scale)


@njit
def _lll(matrix):
    """Pymatgen's column-based LLL, delta=.75, with bounded work.

    Dot shapes, projection subtraction, ties-to-even rounding, and the Lovasz
    expression follow the reference. The 2x2 least-squares problem is actually
    solved: diagonal division is not an equivalent floating-point operation.
    NumPy's rcond=None for that problem is explicitly 2*machine_epsilon.
    """
    basis = matrix.copy().T
    orthogonal = np.zeros((3, 3))
    coefficients = np.zeros((3, 3))
    norms = np.zeros(3)
    near_tie = False
    for i in range(3):
        if i:
            coefficients[i, :i] = np.dot(basis[:, i], orthogonal[:, :i]) / norms[:i]
            orthogonal[:, i] = basis[:, i] - np.dot(orthogonal[:, :i], coefficients[i, :i])
        else:
            orthogonal[:, i] = basis[:, i]
        norms[i] = np.dot(orthogonal[:, i], orthogonal[:, i])
        if not np.isfinite(norms[i]) or norms[i] <= 1e-28:
            return matrix.copy(), False, near_tie

    k = 2
    for iteration in range(_LLL_STEPS):
        if k > 3:
            return basis.T.copy(), True, near_tie
        for i in range(k - 1, 0, -1):
            coefficient = coefficients[k - 1, i - 1]
            if not np.isfinite(coefficient) or abs(coefficient) > 1e8:
                return matrix.copy(), False, near_tie
            # Use the actual BLAS/lstsq result, including either side of a
            # half-integer. Recomputing projections as scalar sums would make
            # this rounding ambiguous on common centred crystal bases.
            near_tie = near_tie or _near(coefficient, math.floor(coefficient) + 0.5, 1.0)
            multiple = round(coefficient)
            if multiple != 0:
                basis[:, k - 1] -= multiple * basis[:, i - 1]
                for j in range(i):
                    previous = 1.0 if j == i - 1 else coefficients[i - 1, j]
                    coefficients[k - 1, j] -= multiple * previous

        left = np.dot(orthogonal[:, k - 1], orthogonal[:, k - 1])
        right = (0.75 - abs(coefficients[k - 1, k - 2]) ** 2) * np.dot(
            orthogonal[:, k - 2], orthogonal[:, k - 2]
        )
        near_tie = near_tie or _near(left, right, max(norms))
        if left >= right:
            k += 1
            continue

        swap = basis[:, k - 1].copy()
        basis[:, k - 1] = basis[:, k - 2]
        basis[:, k - 2] = swap
        for s in range(k - 1, k + 1):
            i = s - 1
            coefficients[i, :i] = np.dot(basis[:, i], orthogonal[:, :i]) / norms[:i]
            orthogonal[:, i] = basis[:, i] - np.dot(orthogonal[:, :i], coefficients[i, :i])
            norms[i] = np.dot(orthogonal[:, i], orthogonal[:, i])
            if not np.isfinite(norms[i]) or norms[i] <= 1e-28:
                return matrix.copy(), False, near_tie
        if k > 2:
            k -= 1
        else:
            if min(norms[:2]) <= 1e-12 * max(norms[:2]):
                return matrix.copy(), False, near_tie
            products = np.dot(basis[:, k:3].T, orthogonal[:, k - 2:k])
            diagonal = np.diag(norms[k - 2:k])
            solution = np.linalg.lstsq(diagonal.T, products.T, rcond=2 * _EPS)[0].T
            coefficients[k:3, k - 2:k] = solution
    return matrix.copy(), False, near_tie


@njit
def _metric_parameters(metric):
    """The reference names A, B, C, E, N, Y in that order."""
    return (metric[0, 0], metric[1, 1], metric[2, 2],
            2 * metric[1, 2], 2 * metric[0, 2], 2 * metric[0, 1])


@njit
def _metric_boundary(A, B, C, E, N, Y, epsilon):
    """Check the actual epsilon boundaries, allowing symmetry ties."""
    scale = max(A, B, C, abs(E), abs(N), abs(Y))
    pairs = (
        (B + epsilon, A), (abs(A - B), epsilon), (abs(E), abs(N) + epsilon),
        (C + epsilon, B), (abs(B - C), epsilon), (abs(N), abs(Y) + epsilon),
        (abs(E), epsilon), (abs(N), epsilon), (abs(Y), epsilon),
        (abs(E), B + epsilon), (abs(E - B), epsilon), (Y - epsilon, 2 * N),
        (abs(E + B), epsilon), (-epsilon, Y),
        (abs(N), A + epsilon), (abs(A - N), epsilon), (Y - epsilon, 2 * E),
        (abs(A + N), epsilon), (abs(Y), A + epsilon), (abs(A - Y), epsilon),
        (N - epsilon, 2 * E), (abs(A + Y), epsilon), (-epsilon, N),
        (-epsilon, E + N + Y + A + B),
        (abs(E + N + Y + A + B), epsilon), (epsilon, Y + (A + N) * 2),
    )
    for left, right in pairs:
        if _near(left, right, scale):
            return True
    return False


@njit
def _niggli_metric(metric, epsilon):
    """A1--A8, including the post-A1 refresh and nested dot order."""
    for iteration in range(_NIGGLI_STEPS):
        A, B, C, E, N, Y = _metric_parameters(metric)
        if _metric_boundary(A, B, C, E, N, Y, epsilon):
            return metric, False
        if B + epsilon < A or (abs(A - B) < epsilon and abs(E) > abs(N) + epsilon):
            change = np.array([[0., -1., 0.], [-1., 0., 0.], [0., 0., -1.]])
            metric = np.dot(change.T, np.dot(metric, change))
            A, B, C, E, N, Y = _metric_parameters(metric)
            if _metric_boundary(A, B, C, E, N, Y, epsilon):
                return metric, False
        if C + epsilon < B or (abs(B - C) < epsilon and abs(N) > abs(Y) + epsilon):
            change = np.array([[-1., 0., 0.], [0., 0., -1.], [0., -1., 0.]])
            metric = np.dot(change.T, np.dot(metric, change))
            continue

        ell = 0.0 if abs(E) < epsilon else E / abs(E)
        em = 0.0 if abs(N) < epsilon else N / abs(N)
        en = 0.0 if abs(Y) < epsilon else Y / abs(Y)
        if ell * em * en == 1:
            i = -1.0 if ell == -1 else 1.0
            j = -1.0 if em == -1 else 1.0
            k = -1.0 if en == -1 else 1.0
        else:
            i = -1.0 if ell == 1 else 1.0
            j = -1.0 if em == 1 else 1.0
            k = -1.0 if en == 1 else 1.0
            if i * j * k == -1:
                if en == 0:
                    k = -1.0
                elif em == 0:
                    j = -1.0
                elif ell == 0:
                    i = -1.0
        change = np.diag(np.array([i, j, k]))
        metric = np.dot(change.T, np.dot(metric, change))
        A, B, C, E, N, Y = _metric_parameters(metric)
        if _metric_boundary(A, B, C, E, N, Y, epsilon):
            return metric, False
        change = np.eye(3)
        if abs(E) > B + epsilon or (abs(E - B) < epsilon and Y - epsilon > 2 * N) or (
            abs(E + B) < epsilon and -epsilon > Y
        ):
            if E == 0:
                return metric, False
            change[1, 2] = -E / abs(E)
        elif abs(N) > A + epsilon or (abs(A - N) < epsilon and Y - epsilon > 2 * E) or (
            abs(A + N) < epsilon and -epsilon > Y
        ):
            if N == 0:
                return metric, False
            change[0, 2] = -N / abs(N)
        elif abs(Y) > A + epsilon or (abs(A - Y) < epsilon and N - epsilon > 2 * E) or (
            abs(A + Y) < epsilon and -epsilon > N
        ):
            if Y == 0:
                return metric, False
            change[0, 1] = -Y / abs(Y)
        elif -epsilon > E + N + Y + A + B or (
            abs(E + N + Y + A + B) < epsilon < Y + (A + N) * 2
        ):
            change[0, 2] = 1.0
            change[1, 2] = 1.0
        else:
            return metric, True
        metric = np.dot(change.T, np.dot(metric, change))
    # Pymatgen proceeds after 100 steps; an exhausted budget is uncertified here.
    return metric, False


@njit
def _target_parameters(metric):
    """Metric -> math.acos degrees -> from_parameters -> lengths/angles.

    The two angle conversions intentionally have different operation orders:
    math.acos(x)/pi*180 before reconstruction, np.arccos(x)*180/pi after it.
    """
    lengths = np.sqrt(np.diag(metric))
    angles = np.empty(3)
    for i in range(3):
        j, k = ((1, 2), (0, 2), (0, 1))[i]
        value = (2 * metric[j, k]) / 2 / lengths[j] / lengths[k]
        if not np.isfinite(value) or abs(value) >= 1:
            return lengths, angles, False
        angles[i] = math.acos(value) / math.pi * 180
    radians = np.radians(angles)
    ca, cb, cg = np.cos(radians)
    sa, sb, unused = np.sin(radians)
    if sa * sb == 0:
        return lengths, angles, False
    value = (ca * cb - cg) / (sa * sb)
    gamma_star = np.arccos(min(1.0, max(-1.0, value)))
    a, b, c = lengths
    target = np.array([
        [a * sb, 0., a * cb],
        [-b * sa * np.cos(gamma_star), b * sa * np.sin(gamma_star), b * ca],
        [0., 0., c],
    ])
    lengths = _lengths(target)
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        value = np.dot(target[j], target[k]) / (lengths[j] * lengths[k])
        angles[i] = np.arccos(min(1.0, max(-1.0, value))) * 180.0 / np.pi
    return lengths, angles, True


@njit
def _sphere_images(matrix, radius):
    """Bounded single-origin specialization of the Cython linked cells.

    Reciprocal cross/dot, the .15 padding, exclusive image upper bound,
    coordinate box, squared-radius tolerance, and cube-index additions keep
    their original order. The radius<1 recursive search is preserved.
    """
    search_radius = 1.0 + 1e-8 if radius < 1.0 else radius
    lower = -search_radius - 1e-8
    upper = search_radius + 1e-8
    ncube = int(math.ceil((upper - lower) / search_radius))
    if ncube != 3:
        return np.empty((0, 3)), np.empty(0), False
    bounds = np.empty(3, dtype=np.int64)
    count = 1
    for i in range(3):
        cross = np.cross(matrix[(i + 1) % 3], matrix[(i + 2) % 3])
        product = _sum3(matrix[i], cross)
        if product == 0 or not np.isfinite(product):
            return np.empty((0, 3)), np.empty(0), False
        reciprocal = (2 * np.pi * cross) / product
        extent = (search_radius + 0.15) * math.sqrt(_sum3(reciprocal, reciprocal)) / (2 * np.pi)
        if not np.isfinite(extent) or extent > _MAX_IMAGES:
            return np.empty((0, 3)), np.empty(0), False
        # Rounding at this ceil boundary changes only redundant outer images:
        # the .15 padding exceeds the 1e-8 sphere/box tolerance in this domain.
        bounds[i] = int(math.ceil(extent)) + 1
        count *= 2 * bounds[i]
        if count > _MAX_IMAGES:
            return np.empty((0, 3)), np.empty(0), False

    images = np.empty((_MAX_CANDIDATES, 3))
    distances = np.empty(_MAX_CANDIDATES)
    links = np.empty(_MAX_CANDIDATES, dtype=np.int64)
    heads = np.full(27, -1, dtype=np.int64)
    used = 0
    cart = np.empty(3)
    for i in range(-bounds[0], bounds[0]):
        for j in range(-bounds[1], bounds[1]):
            for k in range(-bounds[2], bounds[2]):
                for axis in range(3):
                    cart[axis] = (i * matrix[0, axis] + j * matrix[1, axis]) + k * matrix[2, axis] + 0.0
                    if _near(cart[axis], lower, search_radius) or _near(cart[axis], upper, search_radius):
                        return images[:0], distances[:0], False
                if (cart[0] <= lower or cart[0] >= upper or cart[1] <= lower
                        or cart[1] >= upper or cart[2] <= lower or cart[2] >= upper):
                    continue
                squared = _sum3(cart, cart)
                limit = search_radius * search_radius + 1e-8
                if _near(squared, limit, limit):
                    return images[:0], distances[:0], False
                if squared >= limit:
                    continue
                distance = math.sqrt(squared)
                if radius < 1.0 and _near(distance, radius, radius):
                    return images[:0], distances[:0], False
                if radius < 1.0 and distance > radius:
                    continue
                cube = 0
                for axis in range(3):
                    position = (cart[axis] - lower + 1e-8) / search_radius
                    nearest = round(position)
                    if _near(position, nearest, 1.0):
                        return images[:0], distances[:0], False
                    index = int(math.floor(position))
                    if index < 0 or index >= 3:
                        return images[:0], distances[:0], False
                    cube = cube * 3 + index
                if used == _MAX_CANDIDATES:
                    return images[:0], distances[:0], False
                images[used, 0] = i
                images[used, 1] = j
                images[used, 2] = k
                distances[used] = distance
                links[used] = heads[cube]
                heads[cube] = used
                used += 1

    ordered = np.empty((used, 3))
    ordered_distances = np.empty(used)
    position = 0
    for cube in range(27):
        index = heads[cube]
        # Bounded traversal even if this helper is changed later.
        for unused in range(used):
            if index == -1:
                break
            ordered[position] = images[index]
            ordered_distances[position] = distances[index]
            position += 1
            index = links[index]
    return ordered, ordered_distances, True


@njit
def _mapping_angle(a, b, a_length, b_length, target):
    """Separate divisions and atol=1, rtol=0 of find_all_mappings."""
    value = np.dot(a, b) / a_length / b_length
    angle = np.arccos(min(1.0, max(-1.0, value))) * 180.0 / np.pi
    difference = abs(angle - target)
    if _near(difference, 1.0, 180.0):
        return 2
    return 0 if difference <= 1.0 else 1


@njit
def _find_mapping(matrix, lengths, angles, epsilon):
    """First nonsingular matching (a,b,c), then determinant sign flip."""
    radius = max(lengths) * (1 + epsilon)
    images, distances, valid = _sphere_images(matrix, radius)
    if not valid:
        return matrix.copy(), False
    cartesian = np.dot(images, matrix)
    indices = np.empty((3, len(images)), dtype=np.int64)
    counts = np.zeros(3, dtype=np.int64)
    cart_lengths = np.empty(len(images))
    lower = 1 / (1 + epsilon)
    upper = 1 + epsilon
    for q in range(len(images)):
        cart_lengths[q] = math.sqrt(_sum3(cartesian[q], cartesian[q]))
        for axis in range(3):
            ratio = distances[q] / lengths[axis]
            if _near(ratio, lower, 1.0) or _near(ratio, upper, 1.0):
                return matrix.copy(), False
            if lower < ratio < upper:
                indices[axis, counts[axis]] = q
                counts[axis] += 1
    if counts[0] * counts[1] * counts[2] > _MAX_MATCHES:
        return matrix.copy(), False
    for ai in range(counts[0]):
        a = indices[0, ai]
        for bi in range(counts[1]):
            b = indices[1, bi]
            gamma = _mapping_angle(cartesian[a], cartesian[b], cart_lengths[a], cart_lengths[b], angles[2])
            if gamma == 2:
                return matrix.copy(), False
            if gamma == 1:
                continue
            for ci in range(counts[2]):
                c = indices[2, ci]
                alpha = _mapping_angle(cartesian[b], cartesian[c], cart_lengths[b], cart_lengths[c], angles[0])
                beta = _mapping_angle(cartesian[a], cartesian[c], cart_lengths[a], cart_lengths[c], angles[1])
                if alpha == 2 or beta == 2:
                    return matrix.copy(), False
                if alpha == 1 or beta == 1:
                    continue
                transform = np.empty((3, 3))
                transform[0], transform[1], transform[2] = images[a], images[b], images[c]
                if abs(np.linalg.det(transform)) < 1e-8:
                    continue
                result = np.empty((3, 3))
                result[0], result[1], result[2] = cartesian[a], cartesian[b], cartesian[c]
                if np.linalg.det(result) <= 0:
                    result = -result
                return result, True
    return matrix.copy(), False


@njit
def _keep_original(result, matrix):
    """np.allclose(result, input), with the input on the relative side."""
    keep = True
    for i in range(3):
        for j in range(3):
            difference = abs(result[i, j] - matrix[i, j])
            tolerance = 1e-8 + 1e-5 * abs(matrix[i, j])
            if _near(difference, tolerance, tolerance):
                return result, False
            if difference > tolerance:
                keep = False
    return matrix.copy() if keep else result, True


@njit
def _reduce_one(matrix):
    """Guard, reduction, and structure-level keep-original branch."""
    if not np.isfinite(matrix).all():
        return matrix.copy(), 2
    largest = np.max(np.abs(matrix))
    if largest < 1e-8 or largest > 1e6:
        return matrix.copy(), 2
    volume = abs(np.dot(np.cross(matrix[0], matrix[1]), matrix[2]))
    if not np.isfinite(volume) or volume <= 1e-12 * largest ** 3:
        return matrix.copy(), 2
    lll, valid, lll_tie = _lll(matrix)
    if not valid:
        return matrix.copy(), 2
    lengths = _lengths(lll)
    if _near(min(lengths), 0.5, 0.5):
        return matrix.copy(), 2
    safe = min(lengths) >= 0.5
    if safe:
        inverse = np.linalg.inv(matrix)
        lower_images = 1.0
        upper_images = 1.0
        for i in range(3):
            extent = 1.5 * max(lengths) * math.sqrt(_sum3(inverse[:, i], inverse[:, i]))
            nearest = round(extent)
            if _near(extent, nearest, 1.0):
                lower_images *= 2 * nearest + 1
                upper_images *= 2 * (nearest + 1) + 1
            else:
                lower_images *= 2 * math.ceil(extent) + 1
                upper_images *= 2 * math.ceil(extent) + 1
        if (lower_images < 1e6) != (upper_images < 1e6):
            return matrix.copy(), 2
        safe = upper_images < 1e6
    if not safe:
        # LLL alone exposes choices at half-integers/Lovasz equalities. The
        # fixture and live BLAS have disagreed on these tiny symmetric cells;
        # unlike the Niggli path, there is no subsequent canonical mapping.
        if lll_tie:
            return matrix.copy(), 2
        result, valid = _keep_original(lll, matrix)
        return result, 1 if valid else 2

    epsilon = 1e-5 * float(volume) ** (1 / 3)
    metric, valid = _niggli_metric(np.dot(lll, lll.T), epsilon)
    if not valid or min(np.diag(metric)) <= 0 or not np.isfinite(metric).all():
        return matrix.copy(), 2
    lengths, angles, valid = _target_parameters(metric)
    if not valid:
        return matrix.copy(), 2
    result, valid = _find_mapping(matrix, lengths, angles, epsilon)
    if not valid:
        return matrix.copy(), 2
    result, valid = _keep_original(result, matrix)
    return result, 0 if valid else 2


@njit
def _reduce_batch(lattices):
    """Compile the batch loop as well as the small lattice kernels."""
    result = np.empty_like(lattices)
    codes = np.empty(len(lattices), dtype=np.int8)
    for i in range(len(lattices)):
        result[i], codes[i] = _reduce_one(lattices[i])
    return result, codes


def reduce_lattices(L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(B,3,3) float64 row lattices -> (float64 reduced, int8 code).

    Codes 0 and 1 certify bitwise reference parity for Niggli and guard-selected
    LLL respectively. Code 2 means the caller must run its reference fallback;
    that output matrix is undefined. Input is never modified. Invalid batch
    shape/dtype is a caller error, while invalid cells are individually code 2.
    """
    if not isinstance(L, np.ndarray) or L.ndim != 3 or L.shape[1:] != (3, 3):
        raise ValueError("L must have shape (B, 3, 3)")
    if L.dtype != np.float64:
        raise TypeError("L must have dtype float64")
    return _reduce_batch(np.ascontiguousarray(L))
