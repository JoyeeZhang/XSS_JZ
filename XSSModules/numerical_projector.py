import numpy as np

'''========================================================================
[C] - FBFPJosephProjector
[I] - image_shape: image dimensions [nrow,ncol]
[I] - pixel_size: physical pixel spacing; a scalar for square pixels or
      [drow,dcol] for different row and column spacings
[I] - lambda_start, dlambda, nview: view-angle grid definition,
      lambda[k] = lambda_start + k * dlambda
[I] - channel_offset, channel_spacing, nchannel: flat-detector channel grid,
      u[k] = (k - channel_offset) * channel_spacing
[I] - SID: source-to-isocenter distance
[I] - SPD: source-to-piercing_point distance
[I] - beam_upsampling: odd number of beamlets averaged per detector channel
[N] - Joseph forward projector and its matched discrete adjoint for 2-D
      flat-detector fan-beam geometry. Images have size [nrow,ncol].
      Complete sinograms have size [nview,nchannel]. Optional zero-based
      view indices allow the same operator to be used for ordered subsets.
[H] - 20260813 created by JZhang
========================================================================'''
class FBFPJosephProjector:

    def __init__(
        self,
        image_shape,
        pixel_size,
        lambda_start,
        dlambda,
        nview,
        channel_offset,
        channel_spacing,
        nchannel,
        SID,
        SPD,
        beam_upsampling=1,
    ):
        image_shape = np.asarray(image_shape)
        self.nrow = int(image_shape[0])
        self.ncol = int(image_shape[1])

        pixel_size = np.asarray(pixel_size, dtype=np.float64)
        self.drow = float(pixel_size[0])
        self.dcol = float(pixel_size[1])

        self.orow = (self.nrow - 1) / 2.0
        self.ocol = (self.ncol - 1) / 2.0
        self.lambda_start = float(lambda_start)
        self.dlambda = float(dlambda)
        self.nview = int(nview)
        self.channel_offset = float(channel_offset)
        self.channel_spacing = float(channel_spacing)
        self.nchannel = int(nchannel)
        self.SID = float(SID)
        self.SPD = float(SPD)
        self.beam_upsampling = int(beam_upsampling)

        if self.nrow <= 1 or self.ncol <= 1:
            raise ValueError("Joseph interpolation requires at least 2 rows and columns")
        if self.nview <= 0 or self.nchannel <= 0:
            raise ValueError("nview and nchannel must be positive")
        if self.drow <= 0 or self.dcol <= 0 or self.channel_spacing <= 0:
            raise ValueError("pixel and detector-channel spacings must be positive")
        if self.SPD <= 0:
            raise ValueError("SPD must be positive")
        if self.beam_upsampling <= 0 or self.beam_upsampling % 2 != 1:
            raise ValueError("beam_upsampling must be a positive odd integer")

        self.image_shape = (self.nrow, self.ncol)
        self.sinogram_shape = (self.nview, self.nchannel)
        self._x = (np.arange(self.ncol) - self.ocol) * self.dcol
        self._y = (np.arange(self.nrow) - self.orow) * self.drow

    '''--------------------------------------------------------------------
    [F] - read_view_indices
    [I] - views: None or a one-dimensional sequence of zero-based view indices
    [O] - indices: validated np.ndarray of selected view indices
    [N] - The view indices are needed for ordered-subsets reconstruction.
          Each ordered subset passes its own interleaved view indices to both
          forward() and adjoint(). When views is None, all views are selected
          for full-data projection or reconstruction.
    [H] - 20260813 created by JZhang
    --------------------------------------------------------------------'''
    def read_view_indices(self, views):

        if views is None:
            ind_view = np.arange(self.nview, dtype=np.int64)
        else:
            ind_view = np.asarray(views, dtype=np.int64)
            if ind_view.ndim != 1:
                raise ValueError("views must be a one-dimensional sequence")
            if np.any(ind_view < 0) or np.any(ind_view >= self.nview):
                raise IndexError("view index outside the complete view grid")

        return ind_view

    '''--------------------------------------------------------------------
    [F] - forward
    [I] - image: attenuation image, np.ndarray of size [nrow,ncol]
    [I] - views: optional zero-based view indices; None selects all views
    [O] - output: line-integral sinogram of size [nselected,nchannel]
    [N] - Apply the Joseph forward operator A. Each ray is sampled along
          its dominant Cartesian axis using linear interpolation. When
          beam_upsampling > 1, beamlet results are averaged per channel.
    [H] - 20260813 created by JZhang
    --------------------------------------------------------------------'''
    def forward(self, image, views=None):

        image = np.asarray(image)
        if image.shape != self.image_shape:
            raise ValueError(f"image shape must be {self.image_shape}, got {image.shape}")

        ind_view = self.read_view_indices(views)

        nups = self.beam_upsampling
        nchannel_up = nups * self.nchannel
        spacing_up = self.channel_spacing / nups
        offset_up = self.channel_offset * nups + 0.5 * (nups - 1)
        detector_u = (np.arange(nchannel_up) - offset_up) * spacing_up
        gamma = np.arctan(detector_u / self.SPD)
        distance = self.SID * np.sin(gamma)
        output = np.zeros((ind_view.size, self.nchannel), dtype=np.result_type(image, float))
        image_flat = image.ravel()

        for output_view, view in enumerate(ind_view):
            angle = np.pi/2.0 + self.lambda_start + int(view) * self.dlambda - gamma
            ray_values = np.zeros(nups * self.nchannel, dtype=output.dtype)
            sin_angle = np.sin(angle)
            cos_angle = np.cos(angle)

            x_driven = np.abs(sin_angle) >= np.sqrt(0.5)
            for ray in np.flatnonzero(x_driven):
                fractional_row = ((distance[ray] - self._x * cos_angle[ray]) / sin_angle[ray]) / self.drow + self.orow
                lower = np.floor(fractional_row).astype(np.int64)
                valid = (fractional_row >= 0) & (fractional_row < self.nrow - 1)
                columns = np.flatnonzero(valid)
                lower = lower[valid]
                alpha = fractional_row[valid] - lower
                base = lower * self.ncol + columns
                samples = (1.0 - alpha) * image_flat[base] + alpha * image_flat[base + self.ncol]
                ray_values[ray] = samples.sum() * self.dcol / abs(sin_angle[ray])

            for ray in np.flatnonzero(~x_driven):
                fractional_col = ((distance[ray] - self._y * sin_angle[ray]) / cos_angle[ray]) / self.dcol + self.ocol
                lower = np.floor(fractional_col).astype(np.int64)
                valid = (fractional_col >= 0) & (fractional_col < self.ncol - 1)
                rows = np.flatnonzero(valid)
                lower = lower[valid]
                alpha = fractional_col[valid] - lower
                base = rows * self.ncol + lower
                samples = (1.0 - alpha) * image_flat[base] + alpha * image_flat[base + 1]
                ray_values[ray] = samples.sum() * self.drow / abs(cos_angle[ray])

            output[output_view] = ray_values.reshape(self.nchannel, nups).mean(axis=1)

        return output

    '''--------------------------------------------------------------------
    [F] - backward
    [I] - sinogram: ray-domain values, np.ndarray of size
          [nselected,nchannel]
    [I] - views: optional zero-based view indices corresponding exactly to
          the first dimension of sinogram; None selects all views
    [O] - output: backprojected image of size [nrow,ncol]
    [N] - Apply the exact discrete transpose A.T of forward(). This method
          contains no ramp filter or FBP-specific geometrical weighting.
          Linear interpolation is transposed by scattering each ray value
          to the same neighboring pixels with the same weights.
    [H] - 20260813 created by JZhang
    --------------------------------------------------------------------'''
    def backward(self, sinogram, views=None):

        ind_view = self.read_view_indices(views)

        sinogram = np.asarray(sinogram)
        expected_shape = (ind_view.size, self.nchannel)
        if sinogram.shape != expected_shape:
            raise ValueError(f"sinogram shape must be {expected_shape}, got {sinogram.shape}")

        nups = self.beam_upsampling
        nchannel_up = nups * self.nchannel
        spacing_up = self.channel_spacing / nups
        offset_up = self.channel_offset * nups + 0.5 * (nups - 1)
        detector_u = (np.arange(nchannel_up) - offset_up) * spacing_up
        gamma = np.arctan(detector_u / self.SPD)
        distance = self.SID * np.sin(gamma)
        output = np.zeros(self.image_shape, dtype=np.result_type(sinogram, float))
        output_flat = output.ravel()

        for input_view, view in enumerate(ind_view):
            ray_values = np.repeat(sinogram[input_view] / nups, nups)
            angle = np.pi / 2.0 + self.lambda_start + int(view) * self.dlambda - gamma
            sin_angle = np.sin(angle)
            cos_angle = np.cos(angle)

            x_driven = np.abs(sin_angle) >= np.sqrt(0.5)
            for ray in np.flatnonzero(x_driven):
                fractional_row = ((distance[ray] - self._x * cos_angle[ray]) / sin_angle[ray]) / self.drow + self.orow
                lower = np.floor(fractional_row).astype(np.int64)
                valid = (fractional_row >= 0) & (fractional_row < self.nrow - 1)
                columns = np.flatnonzero(valid)
                lower = lower[valid]
                alpha = fractional_row[valid] - lower
                base = lower * self.ncol + columns
                scale = ray_values[ray] * self.dcol / abs(sin_angle[ray])
                np.add.at(output_flat, base, scale * (1.0 - alpha))
                np.add.at(output_flat, base + self.ncol, scale * alpha)

            for ray in np.flatnonzero(~x_driven):
                fractional_col = ((distance[ray] - self._y * sin_angle[ray]) / cos_angle[ray]) / self.dcol + self.ocol
                lower = np.floor(fractional_col).astype(np.int64)
                valid = (fractional_col >= 0) & (fractional_col < self.ncol - 1)
                rows = np.flatnonzero(valid)
                lower = lower[valid]
                alpha = fractional_col[valid] - lower
                base = rows * self.ncol + lower
                scale = ray_values[ray] * self.drow / abs(cos_angle[ray])
                np.add.at(output_flat, base, scale * (1.0 - alpha))
                np.add.at(output_flat, base + 1, scale * alpha)

        return output

    '''--------------------------------------------------------------------
    [F] - adjoint_test
    [I] - views: optional zero-based view indices; None selects all views
    [I] - seed: random-number seed used to generate the test arrays
    [O] - lhs: inner product <A*x,y>
    [O] - rhs: inner product <x,A.T*y>
    [O] - relative_error: normalized absolute difference between lhs, rhs
    [N] - Verify numerically that backward() is the discrete transpose of
          forward(). For float64 calculations, relative_error should be
          close to floating-point precision.
    [H] - 20260813 created by JZhang
    --------------------------------------------------------------------'''
    def adjoint_test(self, views=None, seed=0):

        ind_view = self.read_view_indices(views)
        rng = np.random.default_rng(seed)
        image = rng.standard_normal(self.image_shape)
        sinogram = rng.standard_normal((ind_view.size, self.nchannel))
        lhs = float(np.vdot(self.forward(image, ind_view), sinogram).real)
        rhs = float(np.vdot(image, self.backward(sinogram, ind_view)).real)
        relative_error = abs(lhs - rhs) / max(abs(lhs), abs(rhs), np.finfo(float).eps)

        return lhs, rhs, relative_error
