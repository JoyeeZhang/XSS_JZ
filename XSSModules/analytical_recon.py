import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift
import json

from XSSModules import filter
from XSSModules import file_io as fio
import time
import os

class FBCurveD:
    def __init__(self, projection_header, projection, image_shape, pixel_size, filter_type):
        self.load_recon_protocol(projection_header, image_shape, pixel_size)
        self.load_projection_data(projection)
        self.load_filter(filter_type)

    def load_recon_protocol(self, projection_header, image_shape, pixel_size):
        image_shape = tuple(int(value) for value in image_shape)
        if len(image_shape) != 2 or image_shape[0] != image_shape[1]:
            raise ValueError("image_shape must describe a square 2-D image")

        pixel_size = np.asarray(pixel_size, dtype=np.float64)
        if pixel_size.ndim == 0:
            pixel_size_value = float(pixel_size)
        elif pixel_size.shape == (2,) and np.isclose(pixel_size[0], pixel_size[1]):
            pixel_size_value = float(pixel_size[0])
        else:
            raise ValueError("pixel_size must be a scalar or equal (drow, dcol) values")
        if pixel_size_value <= 0.0:
            raise ValueError("pixel_size must be positive")

        self.nview = int(projection_header["nview"])
        self.dlambda = float(projection_header["angular_space_per_view"])
        self.lambda_start = float(projection_header["starting_angular_position"])
        self.SID = float(projection_header["SID"])
        self.SDD = float(projection_header["SDD"])
        self.nsampling = int(projection_header["nchannel"])
        self.osampling = float(projection_header["ochannel"])
        self.dgamma = float(projection_header["dchannel"]) / self.SDD

        self.img_size = image_shape[0]
        self.pixel_size = pixel_size_value
        self.fov = self.img_size * self.pixel_size

        self.pad_length = 2 ** int(np.ceil(np.log2(2*self.nsampling-1)))

    def load_projection_data(self, projection):
        projection = np.asarray(projection, dtype=np.float64)
        if projection.ndim == 3:
            projection = projection[:, projection.shape[1] // 2, :]

        if projection.shape == (self.nview, self.nsampling):
            projection = projection.T
        elif projection.shape != (self.nsampling, self.nview):
            raise ValueError(
                f"Projection data shape is {projection.shape}; expected "
                f"{(self.nview, self.nsampling)}, {(self.nsampling, self.nview)}, "
                "or a 3-D array with shape (nview, nrow, nchannel)"
            )

        self.data = projection
        return self.data

    def load_filter(self, filter_type):
        if filter_type == "hamming":
            alpha = 0.54
            filter_kernel = filter.kernel_with_raised_cosine_window(self.nsampling, self.dgamma, alpha=alpha)
        elif filter_type == "hann":
            alpha = 0.5
            filter_kernel = filter.kernel_with_raised_cosine_window(self.nsampling, self.dgamma, alpha=alpha)
        elif filter_type == "ramp":
            alpha = 1
            filter_kernel = filter.kernel_with_raised_cosine_window(self.nsampling, self.dgamma, alpha=alpha)
        elif filter_type == "sinc":
            raise NotImplementedError("The sinc filter is not implemented")
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")

        self.filter = filter_kernel

    def filtered_backprojection(self):
        nsampling = self.nsampling
        nview = self.nview
        dgamma = self.dgamma
        dlambda = self.dlambda
        pad_length = self.pad_length
        img_size = self.img_size
        pixel_size = self.pixel_size
        SID = self.SID
        data = self.data
        filter = self.filter

        osampling = self.osampling
        val1 = (osampling - np.arange(nsampling)) * dgamma
        weight1 = np.cos(val1)
        weighted_data = data * weight1[:, np.newaxis]
        data_pad = np.zeros((pad_length, nview))
        data_pad[:nsampling, :] = weighted_data

        val2 = np.arange(-nsampling+1, nsampling) * dgamma
        weight2 = np.ones_like(val2)
        mask = (np.abs(val2) > 1e-12)
        weight2[mask] = (val2[mask] / np.sin(val2[mask])) ** 2
        weighted_filter = filter * weight2
        filter_pad = np.zeros((pad_length, 1))
        filter_pad[:nsampling, 0] = weighted_filter[(nsampling-1):]
        filter_pad[-(nsampling-1):, 0] = weighted_filter[:(nsampling-1)]

        data_pad_fft = fft(data_pad, axis=0)
        filter_pad_fft = fft(filter_pad, axis=0)
        filtered_data_pad = np.real(ifft(data_pad_fft * filter_pad_fft * dgamma, axis=0))
        filtered_data = filtered_data_pad[:nsampling, :]

        view_array = self.lambda_start + np.arange(nview) * dlambda
        img_offset = (img_size - 1) / 2.0
        val = np.arange(img_size) - img_offset
        X0 = pixel_size * val[np.newaxis, :]
        Y0 = pixel_size * val[::-1, np.newaxis]

        result = np.zeros((img_size, img_size), dtype=np.float64)
        for iview, lam in enumerate(view_array):
            p = filtered_data[:, iview]
            sin_lam = np.sin(lam)
            cos_lam = np.cos(lam)

            X_lam = X0 * cos_lam - Y0 * sin_lam
            Y_lam = X0 * sin_lam + Y0 * cos_lam
            T_lam = osampling - np.arctan2(Y_lam, (SID - X_lam)) / dgamma

            T_floor = np.floor(T_lam).astype(np.int64)
            T_ceil = T_floor + 1
            alpha = T_lam - T_floor

            T_floor_clip = np.clip(T_floor, 0, nsampling - 1)
            T_ceil_clip = np.clip(T_ceil, 0, nsampling - 1)

            map = (1 - alpha) * p[T_floor_clip] + alpha * p[T_ceil_clip]

            result += (SID / ((SID - X_lam) ** 2 + Y_lam ** 2)) * map

        result *= dlambda / 2.0

        return result

class FBFPD:
    def __init__(self, projection_header, projection, image_shape, pixel_size, filter_type):
        self.load_recon_protocol(projection_header, image_shape, pixel_size)
        self.load_projection_data(projection)
        self.load_filter(filter_type)

    def load_recon_protocol(self, projection_header, image_shape, pixel_size):
        image_shape = tuple(int(value) for value in image_shape)
        if len(image_shape) != 2 or image_shape[0] != image_shape[1]:
            raise ValueError("image_shape must describe a square 2-D image")

        pixel_size = np.asarray(pixel_size, dtype=np.float64)
        if pixel_size.ndim == 0:
            pixel_size_value = float(pixel_size)
        elif pixel_size.shape == (2,) and np.isclose(pixel_size[0], pixel_size[1]):
            pixel_size_value = float(pixel_size[0])
        else:
            raise ValueError("pixel_size must be a scalar or equal (drow, dcol) values")
        if pixel_size_value <= 0.0:
            raise ValueError("pixel_size must be positive")

        self.nview = int(projection_header["nview"])
        self.dlambda = float(projection_header["angular_space_per_view"])
        self.lambda_start = float(projection_header["starting_angular_position"])
        self.dchannel = float(projection_header["dchannel"])
        self.SID = float(projection_header["SID"])
        self.SDD = float(projection_header["SDD"])
        self.nsampling = int(projection_header["nchannel"])
        self.osampling = float(projection_header["ochannel"])

        self.img_size = image_shape[0]
        self.pixel_size = pixel_size_value
        self.fov = self.img_size * self.pixel_size

        self.pad_length = 2 ** int(np.ceil(np.log2(2*self.nsampling-1)))

    def load_projection_data(self, projection):
        projection = np.asarray(projection, dtype=np.float64)
        if projection.ndim == 3:
            projection = projection[:, projection.shape[1] // 2, :]

        if projection.shape == (self.nview, self.nsampling):
            projection = projection.T
        elif projection.shape != (self.nsampling, self.nview):
            raise ValueError(
                f"Projection data shape is {projection.shape}; expected "
                f"{(self.nview, self.nsampling)}, {(self.nsampling, self.nview)}, "
                "or a 3-D array with shape (nview, nrow, nchannel)"
            )

        self.data = projection
        return self.data

    def load_filter(self, filter_type):
        if filter_type == "hamming":
            alpha = 0.54
            filter_kernel = filter.kernel_with_raised_cosine_window(self.nsampling, self.dchannel, alpha=alpha)
        elif filter_type == "hann":
            alpha = 0.5
            filter_kernel = filter.kernel_with_raised_cosine_window(self.nsampling, self.dchannel, alpha=alpha)
        elif filter_type == "ramp":
            alpha = 1
            filter_kernel = filter.kernel_with_raised_cosine_window(self.nsampling, self.dchannel, alpha=alpha)
        elif filter_type == "sinc":
            raise NotImplementedError("The sinc filter is not implemented")
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")

        self.filter = filter_kernel

    def filtered_backprojection(self):
        nsampling = self.nsampling
        nview = self.nview
        dchannel = self.dchannel
        dlambda = self.dlambda
        pad_length = self.pad_length
        img_size = self.img_size
        pixel_size = self.pixel_size
        SID = self.SID
        SDD = self.SDD
        data = self.data
        filter = self.filter

        osampling = self.osampling
        val1 = (osampling - np.arange(nsampling)) * dchannel
        weight1 = SDD / np.sqrt(SDD ** 2 + val1 ** 2)
        weighted_data = data * weight1[:, np.newaxis]
        data_pad = np.zeros((pad_length, nview))
        data_pad[:nsampling, :] = weighted_data

        filter_pad = np.zeros((pad_length, 1))
        filter_pad[:nsampling, 0] = filter[(nsampling-1):]
        filter_pad[-(nsampling-1):, 0] = filter[:(nsampling-1)]

        data_pad_fft = fft(data_pad, axis=0)
        filter_pad_fft = fft(filter_pad, axis=0)
        filtered_data_pad = np.real(ifft(data_pad_fft * filter_pad_fft * dchannel, axis=0))
        filtered_data = filtered_data_pad[:nsampling, :]

        view_array = self.lambda_start + np.arange(nview, dtype=np.float64) * dlambda
        img_offset = (img_size - 1) / 2.0
        val = np.arange(img_size, dtype=np.float64) - img_offset
        X0 = pixel_size * val[np.newaxis, :]
        Y0 = pixel_size * val[::-1, np.newaxis]

        result = np.zeros((img_size, img_size), dtype=np.float64)
        for iview, lam in enumerate(view_array):
            p = filtered_data[:, iview]
            sin_lam = np.sin(lam)
            cos_lam = np.cos(lam)

            X_lam = X0 * cos_lam - Y0 * sin_lam
            Y_lam = X0 * sin_lam + Y0 * cos_lam
            T_lam = osampling - (SDD * Y_lam / (SID - X_lam)) / dchannel

            T_floor = np.floor(T_lam).astype(np.int64)
            T_ceil = T_floor + 1
            alpha = T_lam - T_floor

            T_floor_clip = np.clip(T_floor, 0, nsampling - 1)
            T_ceil_clip = np.clip(T_ceil, 0, nsampling - 1)

            p_map = (1 - alpha) * p[T_floor_clip] + alpha * p[T_ceil_clip]
            result += (SID * SDD / (SID - X_lam) ** 2) * p_map

        result *= dlambda / 2.0

        return result

class CBFPD:
    def __init__(self, projection_header, projection, image_shape, voxel_size, filter_type):
        self.load_recon_protocol(projection_header, image_shape, voxel_size)
        self.load_projection_data(projection)
        self.load_filter(filter_type)

    def load_recon_protocol(self, projection_header, image_shape, voxel_size):
        image_shape = tuple(int(value) for value in image_shape)
        if len(image_shape) != 3 or any(value <= 0 for value in image_shape):
            raise ValueError("image_shape must contain three positive dimensions")

        voxel_size = np.asarray(voxel_size, dtype=np.float64)
        if voxel_size.ndim == 0:
            dz = dx = dy = float(voxel_size)
        elif voxel_size.shape == (3,):
            dz, dx, dy = (float(value) for value in voxel_size)
        else:
            raise ValueError("voxel_size must be a scalar or a (dz, dx, dy) tuple")
        if dz <= 0.0 or dx <= 0.0 or dy <= 0.0:
            raise ValueError("voxel_size values must be positive")

        self.nview = int(projection_header["nview"])
        self.dlambda = float(projection_header["angular_space_per_view"])
        self.lambda_start = float(projection_header["starting_angular_position"])
        self.dchannel = float(projection_header["dchannel"])
        self.drow = float(projection_header["drow"])
        self.SID = float(projection_header["SID"])
        self.SDD = float(projection_header["SDD"])
        self.nchannel = int(projection_header["nchannel"])
        self.nrow = int(projection_header["nrow"])
        self.ochannel = float(projection_header["ochannel"])
        self.orow = float(projection_header["orow"])

        self.nz, self.nx, self.ny = image_shape
        self.dz, self.dx, self.dy = dz, dx, dy
        self.zfov = self.nz * self.dz
        self.xfov = self.nx * self.dx
        self.yfov = self.ny * self.dy
        self.fov_radius = min(self.xfov, self.yfov) / 2.0

        self.pad_length = 2 ** int(np.ceil(np.log2(2*self.nchannel-1)))

    def load_projection_data(self, projection):
        projection = np.asarray(projection, dtype=np.float64)
        if projection.shape == (self.nview, self.nrow, self.nchannel):
            projection = projection.transpose(1, 2, 0)
        elif projection.shape != (self.nrow, self.nchannel, self.nview):
            raise ValueError(
                f"Projection data shape is {projection.shape}; expected "
                f"{(self.nview, self.nrow, self.nchannel)} or "
                f"{(self.nrow, self.nchannel, self.nview)}"
            )

        self.data = projection
        if self.data.shape != (self.nrow, self.nchannel, self.nview):
            raise ValueError(
                f"Projection data shape is {self.data.shape};"
                f"expected {(self.nrow, self.nchannel, self.nview)}"
            )

        return self.data

    def load_filter(self, filter_type):
        if filter_type == "hamming":
            alpha = 0.54
            filter_kernel = filter.kernel_with_raised_cosine_window(self.nchannel, self.dchannel, alpha=alpha)
        elif filter_type == "hann":
            alpha = 0.5
            filter_kernel = filter.kernel_with_raised_cosine_window(self.nchannel, self.dchannel, alpha=alpha)
        elif filter_type == "ramp":
            alpha = 1
            filter_kernel = filter.kernel_with_raised_cosine_window(self.nchannel, self.dchannel, alpha=alpha)
        elif filter_type == "sinc":
            raise NotImplementedError("The sinc filter is not implemented")
        else:
            raise ValueError(f"Unsupported filter type: {filter_type}")

        self.filter = filter_kernel

    def cosine_weight_projections(self):
        ochannel = self.ochannel
        orow = self.orow

        u = (ochannel - np.arange(self.nchannel, dtype=np.float64)) * self.dchannel
        v = (orow - np.arange(self.nrow, dtype=np.float64)) * self.drow

        uu = u[np.newaxis, :]
        vv = v[:, np.newaxis]

        weight = self.SDD / np.sqrt(self.SDD ** 2 + uu ** 2 + vv ** 2)

        return weight


    def filtered_backprojection(self):
        nchannel = self.nchannel
        nrow = self.nrow
        ochannel = self.ochannel
        orow = self.orow
        dchannel = self.dchannel
        drow = self.drow
        nview = self.nview
        dlambda = self.dlambda
        pad_length = self.pad_length
        nx = self.nx
        ny = self.ny
        nz = self.nz
        dx = self.dx
        dy = self.dy
        dz = self.dz
        SID = self.SID
        SDD = self.SDD
        data = self.data
        filter = self.filter

        weight = self.cosine_weight_projections()
        weighted_data = data * weight[:, :, np.newaxis]
        data_pad = np.zeros((nrow, pad_length, nview), dtype=np.float64)
        data_pad[:, :nchannel, :] = weighted_data

        filter_pad = np.zeros((1, pad_length, 1), dtype=np.float64)
        filter_pad[0, :nchannel, 0] = filter[(nchannel-1):]
        filter_pad[0, -(nchannel-1):, 0] = filter[:(nchannel-1)]

        data_pad_fft = fft(data_pad, axis=1)
        filter_pad_fft = fft(filter_pad, axis=1)
        filtered_data_pad = np.real(ifft(data_pad_fft * filter_pad_fft * dchannel, axis=1))
        filtered_data = filtered_data_pad[:, :nchannel, :]

        view_array = self.lambda_start + np.arange(nview, dtype=np.float64) * dlambda
        xoff = (nx - 1) / 2.0
        yoff = (ny - 1) / 2.0
        zoff = (nz - 1) / 2.0
        x = np.arange(nx, dtype=np.float64) - xoff
        y = np.arange(ny, dtype=np.float64) - yoff
        z = np.arange(nz, dtype=np.float64) - zoff
        X0 = dx * x[np.newaxis, :]
        Y0 = dy * y[::-1, np.newaxis]
        Z0 = dz * z

        result = np.zeros((nz, nx, ny), dtype=np.float64)
        start_time = time.perf_counter()
        for iview, lam in enumerate(view_array):
            completed = iview + 1
            if completed == 1 or completed % 5 == 0 or completed == nview:
                elapsed = time.perf_counter() - start_time
                seconds_per_view = elapsed / completed
                remaining = seconds_per_view * (nview - completed)
                print(
                    f"FDK: {completed}/{nview} "
                    f"({100.0 * completed / nview:.1f}%), "
                    f"elapsed={elapsed / 60:.1f} min, "
                    f"remaining={remaining / 60:.1f} min"
                )

            p = filtered_data[:, :, iview]
            sin_lam = np.sin(lam)
            cos_lam = np.cos(lam)

            X_lam = X0 * cos_lam - Y0 * sin_lam
            Y_lam = X0 * sin_lam + Y0 * cos_lam

            T_lam = ochannel - (SDD * Y_lam / (SID - X_lam)) / dchannel
            T_floor = np.floor(T_lam).astype(np.int64)
            T_ceil = T_floor + 1
            alpha_t = T_lam - T_floor

            for i, iz in enumerate(Z0):
                R_lam = orow - (SDD * iz / (SID - X_lam)) / drow
                R_floor = np.floor(R_lam).astype(np.int64)
                R_ceil = R_floor + 1
                alpha_r = R_lam - R_floor

                valid = ((T_floor >= 0) & (T_ceil < nchannel) & (R_floor >= 0) & (R_ceil < nrow))

                p_map = np.zeros((nx, ny), dtype=np.float64)
                if np.any(valid):
                    tf = T_floor[valid]
                    tc = T_ceil[valid]
                    rf = R_floor[valid]
                    rc = R_ceil[valid]
                    at = alpha_t[valid]
                    ar =  alpha_r[valid]

                    p00 = p[rf, tf]
                    p01 = p[rf, tc]
                    p10 = p[rc, tf]
                    p11 = p[rc, tc]

                    map_floor_row = (1 - at) * p00 + at * p01
                    map_ceil_row = (1 - at) * p10 + at * p11

                    p_map[valid] = (1 - ar) * map_floor_row + ar * map_ceil_row

                result[i] += (SID * SDD / (SID - X_lam) ** 2) * p_map

        result *= dlambda / 2.0

        return result
