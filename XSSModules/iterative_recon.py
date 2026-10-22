import numpy as np
import os
from scipy.optimize import Bounds, minimize

class PWLSReconstructionL2GD:

    def __init__(self,
                 line_integrals,
                 statistical_weights,
                 beta,
                 projector,
                 num_iters=50,
                 nonnegative=True,
                 objective_tolerance=1e-6):

        self.line_integrals = np.asarray(line_integrals, dtype=np.float64)
        self.statistical_weights = np.asarray(
            statistical_weights,
            dtype=np.float64,
        )
        self.beta = float(beta)
        self.projector = projector
        self.num_iters = int(num_iters)
        self.nonnegative = bool(nonnegative)
        self.objective_tolerance = float(objective_tolerance)

        if self.line_integrals.shape != self.statistical_weights.shape:
            raise ValueError(
                'line_integrals and statistical_weights must have the same shape'
            )
        if np.any(self.statistical_weights < 0.0):
            raise ValueError('statistical_weights must be nonnegative')
        if self.beta < 0.0:
            raise ValueError('beta must be nonnegative')
        if self.num_iters < 0:
            raise ValueError('num_iters must be nonnegative')
        if self.objective_tolerance < 0.0:
            raise ValueError('objective_tolerance must be nonnegative')

    def objective_function(self, image):
        projection = self.projector.forward(image)
        residual = projection - self.line_integrals
        data_fidelity = 0.5 * np.sum(self.statistical_weights * residual ** 2)
        regularizer = self.beta * 0.5 * np.sum(image ** 2)
        value = data_fidelity + regularizer

        return value

    def gradient(self, image):
        projection = self.projector.forward(image)
        residual = projection - self.line_integrals
        data_gradient = self.projector.backward(self.statistical_weights * residual)
        reg_gradient = self.beta * image
        gradient = data_gradient + reg_gradient

        return gradient

    def hessian_vector_product(self, vector):
        data_hessian = self.projector.backward(
            self.statistical_weights * self.projector.forward(vector)
        )
        reg_hessian = self.beta * vector
        hessian = data_hessian + reg_hessian

        return hessian

    def exact_line_search_step_size(self, gradient):
        numerator = np.sum(gradient ** 2)
        if numerator == 0.0:
            return 0.0

        hessian_gradient = self.hessian_vector_product(gradient)
        denominator = np.sum(gradient * hessian_gradient)
        if denominator <= 0.0:
            raise ValueError(
                'The objective curvature along the gradient must be positive'
            )
        step_size = numerator / denominator

        return step_size

    def reconstruction(self, initial_image, verbose=True, callback=None):
        image = np.asarray(initial_image, dtype=np.float64).copy()

        initial_objective = self.objective_function(image)
        previous_objective = initial_objective
        if verbose:
            print(f'Iteration 0: objective = {initial_objective:.9e}')
        if callback is not None:
            callback(0, image.copy(), initial_objective)

        for iteration in range(1, self.num_iters + 1):
            objective_gradient = self.gradient(image)
            step_size = self.exact_line_search_step_size(objective_gradient)
            if step_size == 0.0:
                if verbose:
                    print(f'Converged before iteration {iteration}: zero gradient')
                break

            image -= step_size * objective_gradient

            if self.nonnegative:
                np.maximum(image, 0.0, out=image)

            current_objective = self.objective_function(image)
            objective_scale = max(
                abs(previous_objective),
                np.finfo(np.float64).eps,
            )
            relative_objective_change = (
                abs(current_objective - previous_objective) / objective_scale
            )
            if verbose:
                print(
                    f'Iteration {iteration}: objective = '
                    f'{current_objective:.9e}, '
                    f'step size = {step_size:.9e}, '
                )
            if callback is not None:
                callback(iteration, image.copy(), current_objective)

            if relative_objective_change <= self.objective_tolerance:
                if verbose:
                    print(
                        f'Converged at iteration {iteration}: '
                        f'relative objective change '
                        f'{relative_objective_change:.9e} <= '
                        f'tolerance {self.objective_tolerance:.9e}'
                    )
                break

            previous_objective = current_objective

        return image


class PWLSReconstructionL2CG(PWLSReconstructionL2GD):

    def __init__(self,
                 line_integrals,
                 statistical_weights,
                 beta,
                 projector,
                 num_iters=50,
                 nonnegative=True,
                 objective_tolerance=1e-6,
                 residual_tolerance=1e-8):
        super().__init__(
            line_integrals=line_integrals,
            statistical_weights=statistical_weights,
            beta=beta,
            projector=projector,
            num_iters=num_iters,
            nonnegative=nonnegative,
            objective_tolerance=objective_tolerance,
        )
        self.residual_tolerance = float(residual_tolerance)

    def negative_projected_gradient(self, image):
        residual = -self.gradient(image)
        if self.nonnegative:
            active = (image <= 0.0) & (residual < 0.0)
            residual[active] = 0.0

        return residual

    def reconstruction(self, initial_image, verbose=True, callback=None):
        image = np.asarray(initial_image, dtype=np.float64).copy()
        if self.nonnegative:
            np.maximum(image, 0.0, out=image)

        previous_objective = self.objective_function(image)
        residual = self.negative_projected_gradient(image)
        direction = residual.copy()
        residual_norm_squared = float(np.sum(residual ** 2))
        initial_residual_norm = np.sqrt(residual_norm_squared)

        if verbose:
            print(f'Iteration 0: objective = {previous_objective:.9e}')
        if callback is not None:
            callback(0, image.copy(), previous_objective)

        for iteration in range(1, self.num_iters + 1):
            residual_norm = np.sqrt(residual_norm_squared)
            residual_scale = max(initial_residual_norm,
                                 np.finfo(np.float64).eps)
            relative_residual = residual_norm / residual_scale
            if relative_residual <= self.residual_tolerance:
                if verbose:
                    print(
                        f'Converged before iteration {iteration}: relative '
                        f'residual {relative_residual:.9e} <= tolerance '
                        f'{self.residual_tolerance:.9e}'
                    )
                break

            hessian_direction = self.hessian_vector_product(direction)
            curvature = float(np.sum(direction * hessian_direction))
            if curvature <= 0.0:
                raise ValueError(
                    'The objective curvature along the CG direction must be '
                    'positive'
                )

            step_size = residual_norm_squared / curvature
            candidate = image + step_size * direction
            projection_activated = False
            if self.nonnegative:
                projection_activated = bool(np.any(candidate < 0.0))
                np.maximum(candidate, 0.0, out=candidate)

            current_objective = self.objective_function(candidate)
            new_residual = self.negative_projected_gradient(candidate)
            new_residual_norm_squared = float(np.sum(new_residual ** 2))

            if projection_activated:
                # Projection invalidates the standard CG recurrence.
                new_direction = new_residual.copy()
            else:
                conjugate_coefficient = (
                    new_residual_norm_squared / residual_norm_squared
                )
                new_direction = (
                    new_residual + conjugate_coefficient * direction
                )

            objective_scale = max(abs(previous_objective),
                                  np.finfo(np.float64).eps)
            relative_objective_change = (
                abs(current_objective - previous_objective) / objective_scale
            )

            image = candidate
            residual = new_residual
            direction = new_direction
            residual_norm_squared = new_residual_norm_squared

            if verbose:
                print(
                    f'Iteration {iteration}: objective = '
                    f'{current_objective:.9e}, residual norm = '
                    f'{np.sqrt(residual_norm_squared):.9e}'
                )
            if callback is not None:
                callback(iteration, image.copy(), current_objective)

            if relative_objective_change <= self.objective_tolerance:
                if verbose:
                    print(
                        f'Converged at iteration {iteration}: relative '
                        f'objective change {relative_objective_change:.9e} '
                        f'<= tolerance {self.objective_tolerance:.9e}'
                    )
                break

            previous_objective = current_objective

        return image


class PWLSReconstructionL2LBFGS(PWLSReconstructionL2GD):

    def __init__(self,
                 line_integrals,
                 statistical_weights,
                 beta,
                 projector,
                 num_iters=50,
                 nonnegative=True,
                 objective_tolerance=1e-6,
                 gradient_tolerance=1e-5,
                 history_size=10,
                 max_line_search=20):
        super().__init__(
            line_integrals=line_integrals,
            statistical_weights=statistical_weights,
            beta=beta,
            projector=projector,
            num_iters=num_iters,
            nonnegative=nonnegative,
            objective_tolerance=objective_tolerance,
        )
        self.gradient_tolerance = float(gradient_tolerance)
        self.history_size = int(history_size)
        self.max_line_search = int(max_line_search)
        self.optimization_result = None

        if self.gradient_tolerance < 0.0:
            raise ValueError('gradient_tolerance must be nonnegative')
        if self.history_size < 1:
            raise ValueError('history_size must be at least 1')
        if self.max_line_search < 1:
            raise ValueError('max_line_search must be at least 1')

    def reconstruction(self, initial_image, verbose=True, callback=None):
        image = np.asarray(initial_image, dtype=np.float64).copy()
        image_shape = tuple(self.projector.image_shape)
        if image.shape != image_shape:
            raise ValueError(
                'initial_image shape must match projector.image_shape: '
                f'{image.shape} != {image_shape}'
            )
        if self.nonnegative:
            np.maximum(image, 0.0, out=image)

        initial_objective = self.objective_function(image)
        if verbose:
            print(f'Iteration 0: objective = {initial_objective:.9e}')
        if callback is not None:
            callback(0, image.copy(), initial_objective)

        def objective_and_gradient(flat_image):
            current_image = flat_image.reshape(image_shape)
            projection = self.projector.forward(current_image)
            residual = projection - self.line_integrals
            objective = (
                0.5 * np.sum(self.statistical_weights * residual ** 2)
                + 0.5 * self.beta * np.sum(current_image ** 2)
            )
            gradient = self.projector.backward(
                self.statistical_weights * residual
            ) + self.beta * current_image

            return float(objective), np.asarray(
                gradient,
                dtype=np.float64,
            ).ravel()

        iteration = 0

        def iteration_callback(flat_image):
            nonlocal iteration
            iteration += 1
            current_image = np.asarray(flat_image).reshape(image_shape)
            current_objective = self.objective_function(current_image)
            if verbose:
                print(
                    f'Iteration {iteration}: objective = '
                    f'{current_objective:.9e}'
                )
            if callback is not None:
                callback(iteration, current_image.copy(), current_objective)

        bounds = Bounds(0.0, np.inf) if self.nonnegative else None
        result = minimize(
            objective_and_gradient,
            image.ravel(),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            callback=iteration_callback,
            options={
                'maxiter': self.num_iters,
                'ftol': self.objective_tolerance,
                'gtol': self.gradient_tolerance,
                'maxcor': self.history_size,
                'maxls': self.max_line_search,
            },
        )
        self.optimization_result = result

        if verbose:
            print(f'L-BFGS-B termination: {result.message}')

        return result.x.reshape(image_shape)


class PWLSReconstructionL1FISTA:
    """PWLS reconstruction with an image L1 penalty using FISTA.

    The optimized objective is

    ``0.5 * ||W^(1/2) (A x - y)||_2^2 + beta * ||x||_1``.

    Backtracking determines a stable gradient step, the L1 term is handled by
    soft thresholding, and Nesterov momentum accelerates the proximal updates.
    """

    def __init__(self,
                 line_integrals,
                 statistical_weights,
                 beta,
                 projector,
                 num_iters=50,
                 nonnegative=True,
                 objective_tolerance=1e-6,
                 initial_step_size=1.0,
                 backtracking_factor=0.5,
                 max_backtracking=30):
        self.line_integrals = np.asarray(line_integrals, dtype=np.float64)
        self.statistical_weights = np.asarray(
            statistical_weights,
            dtype=np.float64,
        )
        self.beta = float(beta)
        self.projector = projector
        self.num_iters = int(num_iters)
        self.nonnegative = bool(nonnegative)
        self.objective_tolerance = float(objective_tolerance)
        self.initial_step_size = float(initial_step_size)
        self.backtracking_factor = float(backtracking_factor)
        self.max_backtracking = int(max_backtracking)

        if self.line_integrals.shape != self.statistical_weights.shape:
            raise ValueError(
                'line_integrals and statistical_weights must have the same shape'
            )
        if np.any(self.statistical_weights < 0.0):
            raise ValueError('statistical_weights must be nonnegative')
        if self.beta < 0.0:
            raise ValueError('beta must be nonnegative')
        if self.num_iters < 0:
            raise ValueError('num_iters must be nonnegative')
        if self.objective_tolerance < 0.0:
            raise ValueError('objective_tolerance must be nonnegative')
        if self.initial_step_size <= 0.0:
            raise ValueError('initial_step_size must be positive')
        if not 0.0 < self.backtracking_factor < 1.0:
            raise ValueError('backtracking_factor must lie between 0 and 1')
        if self.max_backtracking < 1:
            raise ValueError('max_backtracking must be at least 1')

    def data_fidelity(self, image):
        projection = self.projector.forward(image)
        residual = projection - self.line_integrals

        return 0.5 * np.sum(self.statistical_weights * residual ** 2)

    def objective_function(self, image):
        data_fidelity = self.data_fidelity(image)
        regularizer = self.beta * np.sum(np.abs(image))

        return data_fidelity + regularizer

    def data_gradient(self, image):
        projection = self.projector.forward(image)
        residual = projection - self.line_integrals

        return self.projector.backward(self.statistical_weights * residual)

    def proximal_update(self, image, gradient, step_size):
        gradient_update = image - step_size * gradient
        threshold = step_size * self.beta

        if self.nonnegative:
            return np.maximum(gradient_update - threshold, 0.0)

        return np.sign(gradient_update) * np.maximum(
            np.abs(gradient_update) - threshold,
            0.0,
        )

    def backtracking_update(self, image, step_size):
        """Find a proximal update satisfying the smooth descent bound."""
        current_data_fidelity = self.data_fidelity(image)
        gradient = self.data_gradient(image)

        for _ in range(self.max_backtracking):
            candidate = self.proximal_update(image, gradient, step_size)
            difference = candidate - image
            upper_bound = (
                current_data_fidelity
                + np.sum(gradient * difference)
                + np.sum(difference ** 2) / (2.0 * step_size)
            )
            candidate_data_fidelity = self.data_fidelity(candidate)
            rounding_tolerance = (
                10.0 * np.finfo(np.float64).eps
                * max(
                    1.0,
                    abs(current_data_fidelity),
                    abs(candidate_data_fidelity),
                )
            )
            if candidate_data_fidelity <= upper_bound + rounding_tolerance:
                candidate_objective = (
                    candidate_data_fidelity
                    + self.beta * np.sum(np.abs(candidate))
                )
                return candidate, candidate_objective, step_size

            step_size *= self.backtracking_factor

        raise RuntimeError(
            'FISTA backtracking failed; decrease initial_step_size or '
            'increase max_backtracking'
        )

    def reconstruction(self, initial_image, verbose=True, callback=None):
        """Reconstruct an image using accelerated soft thresholding.

        Parameters
        ----------
        initial_image : numpy.ndarray
            Initial image with shape ``projector.image_shape``.
        verbose : bool, optional
            Display the objective value and step size at each iteration.
        callback : callable, optional
            Called as ``callback(iteration, image, objective_value)``.

        Returns
        -------
        numpy.ndarray
            Reconstructed image.
        """
        image = np.asarray(initial_image, dtype=np.float64).copy()
        image_shape = tuple(self.projector.image_shape)
        if image.shape != image_shape:
            raise ValueError(
                'initial_image shape must match projector.image_shape: '
                f'{image.shape} != {image_shape}'
            )
        if self.nonnegative:
            np.maximum(image, 0.0, out=image)

        extrapolated_image = image.copy()
        momentum_parameter = 1.0
        previous_objective = self.objective_function(image)
        step_size = self.initial_step_size

        if verbose:
            print(f'Iteration 0: objective = {previous_objective:.9e}')
        if callback is not None:
            callback(0, image.copy(), previous_objective)

        for iteration in range(1, self.num_iters + 1):
            new_image, current_objective, step_size = self.backtracking_update(
                extrapolated_image,
                step_size,
            )

            # Restart acceleration when extrapolation increases the objective.
            if current_objective > previous_objective:
                momentum_parameter = 1.0
                new_image, current_objective, step_size = (
                    self.backtracking_update(image, step_size)
                )

            new_momentum_parameter = (
                1.0 + np.sqrt(1.0 + 4.0 * momentum_parameter ** 2)
            ) / 2.0
            momentum = (
                (momentum_parameter - 1.0) / new_momentum_parameter
            )
            new_extrapolated_image = (
                new_image + momentum * (new_image - image)
            )

            objective_scale = max(
                abs(previous_objective),
                np.finfo(np.float64).eps,
            )
            relative_objective_change = (
                abs(current_objective - previous_objective) / objective_scale
            )

            image = new_image
            extrapolated_image = new_extrapolated_image
            momentum_parameter = new_momentum_parameter

            if verbose:
                print(
                    f'Iteration {iteration}: objective = '
                    f'{current_objective:.9e}, '
                    f'step size = {step_size:.9e}'
                )
            if callback is not None:
                callback(iteration, image.copy(), current_objective)

            if relative_objective_change <= self.objective_tolerance:
                if verbose:
                    print(
                        f'Converged at iteration {iteration}: relative '
                        f'objective change {relative_objective_change:.9e} '
                        f'<= tolerance {self.objective_tolerance:.9e}'
                    )
                break

            previous_objective = current_objective

        return image


class SSQuadraticPenalty:

    def __init__(self,
                 w_horizontal=1.0,
                 w_vertical=1.0):
        self.w_horizontal = w_horizontal
        self.w_vertical = w_vertical

    def penalty_value(self, image):
        diff_x = image[:, 1:] - image[:, :-1]
        diff_y = image[1:, :] - image[:-1, :]

        penalty_x = 0.5 * self.w_horizontal * np.sum(diff_x**2)
        penalty_y = 0.5 * self.w_vertical * np.sum(diff_y**2)

        value = penalty_x + penalty_y

        return value

    def first_derivative(self, image):
        dR1 = np.zeros_like(image)

        diff_x = image[:, 1:] - image[:, :-1]
        dR1[:, :-1] -= self.w_horizontal * diff_x
        dR1[:, 1:] += self.w_horizontal * diff_x

        diff_y = image[1:, :] - image[:-1, :]
        dR1[:-1, :] -= self.w_vertical * diff_y
        dR1[1:, :] += self.w_vertical * diff_y

        return dR1

    def second_derivative(self, image):
        dR2 = np.zeros_like(image)

        dR2[:, :-1] += 2.0 * self.w_horizontal
        dR2[:, 1:] += 2.0 * self.w_horizontal

        dR2[:-1, :] += 2.0 * self.w_vertical
        dR2[1:, :] += 2.0 * self.w_vertical

        return dR2


class SSPWLSReconstruction:

    def __init__(self,
                 line_integrals,
                 statistical_weights,
                 beta,
                 projector,
                 penalty,
                 num_subsets=1,
                 num_iters=50,
                 nonnegative=True,
                 eps=1e-8,
                 ):
        self.line_integrals = np.asarray(line_integrals)
        self.statistical_weights = np.asarray(statistical_weights)
        self.beta = beta
        self.projector = projector
        self.penalty = penalty
        self.num_subsets = num_subsets
        self.num_iters = num_iters
        self.nonnegative = nonnegative
        self.eps = eps

        self.num_views = self.line_integrals.shape[0]
        if self.line_integrals.shape != self.statistical_weights.shape:
            raise ValueError(
                "line_integrals and statistical_weights "
                "must have the same shape."
            )
        if not (1 <= num_subsets <= self.num_views):
            raise ValueError(
                f"num_subsets must be in the range "
                f"[1, {self.num_views}], "
                f"but got {num_subsets}."
            )

        self.subsets = self.create_ordered_subsets()

    def create_ordered_subsets(self):
        subsets = []

        for i in range(self.num_subsets):
            view_ind = np.arange(i, self.num_views, self.num_subsets)
            subsets.append(view_ind)

        return subsets

    def surrogate_object_function(self, image):
        # surrogate likelihood function
        projection = self.projector.forward(image)
        residual = projection - self.line_integrals
        w = self.statistical_weights
        likelihood_value = 0.5 * np.sum(w * residual**2)

        # surrogate penalty function
        penalty_value = self.penalty.penalty_value(image)

        value = likelihood_value + self.beta * penalty_value

        return value

    def total_first_derivative(self, image, subset_id):
        view_ind = self.subsets[subset_id]

        # first derivative of surrogate likelihood function
        projection = self.projector.forward(image, view_ind)
        residual = projection - self.line_integrals[view_ind]
        w = self.statistical_weights[view_ind]
        dL1 = self.projector.backward(w * residual, view_ind)

        # first derivative of surrogate penalty function
        dR1 = self.penalty.first_derivative(image)

        d1 = self.num_subsets * dL1 + self.beta * dR1

        return d1

    def total_second_derivative(self, image, subset_id):
        view_ind = self.subsets[subset_id]

        # second derivative of surrogate likelihood function
        ones_image = np.ones(image.shape, dtype=np.float64)
        gamma = self.projector.forward(ones_image, view_ind)
        w = self.statistical_weights[view_ind]
        dL2 = self.projector.backward(w * gamma, view_ind)

        # second derivative of surrogate penalty function
        dR2 = self.penalty.second_derivative(image)

        d2 = self.num_subsets * dL2 + self.beta * dR2
        d2 = np.maximum(d2, self.eps)

        return d2

    def reconstruction(self, initial_image, verbose=True, callback=None):
        image = np.asarray(initial_image, dtype=np.float64).copy()
        eps = self.eps
        num_iter = self.num_iters

        if verbose or callback is not None:
            initial_objective = self.surrogate_object_function(image)
        if verbose:
            print(f"Iteration 0: objective = {initial_objective:.9e}")
        if callback is not None:
            callback(0, image.copy(), initial_objective)

        for iteration in range(1, num_iter + 1):
            for subset_id in range(self.num_subsets):
                gradient = self.total_first_derivative(image, subset_id)
                denominator = self.total_second_derivative(image, subset_id)
                valid = denominator > eps
                image[valid] -= gradient[valid] / denominator[valid]
                if self.nonnegative:
                    np.maximum(image, 0.0, out=image)

            if verbose or callback is not None:
                current_objective = self.surrogate_object_function(image)
            if verbose:
                print(
                    f"Iteration {iteration}: objective = "
                    f"{current_objective:.9e}"
                )

            if callback is not None:
                callback(iteration, image.copy(), current_objective)

        return image
