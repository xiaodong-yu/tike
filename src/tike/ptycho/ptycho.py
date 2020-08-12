#!/usr/bin/env python
# -*- coding: utf-8 -*-

# #########################################################################
# Copyright (c) 2018, UChicago Argonne, LLC. All rights reserved.    #
#                                                                         #
# Copyright 2018. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# #########################################################################

__author__ = "Doga Gursoy, Daniel Ching"
__copyright__ = "Copyright (c) 2018, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = [
    "gaussian",
    "reconstruct",
    "simulate",
]

import logging
import numpy as np

from tike.operators import Ptycho
from tike.pool import ThreadPool
from tike.ptycho import solvers
from .position import check_allowed_positions, get_padded_object

logger = logging.getLogger(__name__)


def gaussian(size, rin=0.8, rout=1.0):
    """Return a complex gaussian probe distribution.

    Illumination probe represented on a 2D regular grid.

    A finite-extent circular shaped probe is represented as
    a complex wave. The intensity of the probe is maximum at
    the center and damps to zero at the borders of the frame.

    Parameters
    ----------
    size : int
        The side length of the distribution
    rin : float [0, 1) < rout
        The inner radius of the distribution where the dampening of the
        intensity will start.
    rout : float (0, 1] > rin
        The outer radius of the distribution where the intensity will reach
        zero.

    """
    r, c = np.mgrid[:size, :size] + 0.5
    rs = np.sqrt((r - size / 2)**2 + (c - size / 2)**2)
    rmax = np.sqrt(2) * 0.5 * rout * rs.max() + 1.0
    rmin = np.sqrt(2) * 0.5 * rin * rs.max()
    img = np.zeros((size, size), dtype='float32')
    img[rs < rmin] = 1.0
    img[rs > rmax] = 0.0
    zone = np.logical_and(rs > rmin, rs < rmax)
    img[zone] = np.divide(rmax - rs[zone], rmax - rmin)
    return img


def simulate(
        detector_shape,
        probe, scan,
        psi,
        **kwargs
):  # yapf: disable
    """Return real-valued detector counts of simulated ptychography data."""
    assert scan.ndim == 3
    assert psi.ndim == 3
    check_allowed_positions(scan, psi, probe)
    with Ptycho(
            probe_shape=probe.shape[-1],
            detector_shape=int(detector_shape),
            nz=psi.shape[-2],
            n=psi.shape[-1],
            ntheta=scan.shape[0],
            **kwargs,
    ) as operator:
        data = 0
        for mode in np.split(probe, probe.shape[-3], axis=-3):
            farplane = operator.fwd(
                probe=operator.asarray(mode, dtype='complex64'),
                scan=operator.asarray(scan, dtype='float32'),
                psi=operator.asarray(psi, dtype='complex64'),
                **kwargs,
            )
            data += np.square(
                np.linalg.norm(
                    farplane.reshape(operator.ntheta,
                                     scan.shape[-2] // operator.fly, -1,
                                     detector_shape, detector_shape),
                    ord=2,
                    axis=2,
                ))
        return operator.asnumpy(data)

def reconstruct(
        data,
        probe, scan,
        algorithm,
        psi=None, num_gpu=1, num_tile=1, num_iter=1, rtol=-1, **kwargs
):  # yapf: disable
    """Solve the ptychography problem using the given `algorithm`.

    Parameters
    ----------
    algorithm : string
        The name of one algorithms from :py:mod:`.ptycho.solvers`.
    rtol : float
        Terminate early if the relative decrease of the cost function is
        less than this amount.

    """
    (psi, scan) = get_padded_object(scan, probe) if psi is None else (psi, scan)
    check_allowed_positions(scan, psi, probe)
    if algorithm in solvers.__all__:
        # Initialize an operator.
        with Ptycho(
                probe_shape=probe.shape[-1],
                detector_shape=data.shape[-1],
                nz=psi.shape[-2],
                n=psi.shape[-1],
                ntheta=scan.shape[0],
                **kwargs,
        ) as operator, ThreadPool(num_gpu) as pool:
            logger.info("{} for {:,d} - {:,d} by {:,d} frames for {:,d} "
                        "iterations.".format(algorithm, *data.shape[1:],
                                             num_iter))
            # TODO: Merge code paths num_gpu is not used.
            #num_gpu = pool.device_count
            # send any array-likes to device
            if (num_gpu <= 1):
                data = operator.asarray(data, dtype='float32')
                result = {
                    'psi': operator.asarray(psi, dtype='complex64'),
                    'probe': operator.asarray(probe, dtype='complex64'),
                    'scan': operator.asarray(scan, dtype='float32'),
                }
                for key, value in kwargs.items():
                    if np.ndim(value) > 0:
                        kwargs[key] = operator.asarray(value)
            else:
                scan, data = asarray_multi_split(
                    operator,
                    num_gpu,
                    num_tile,
                    scan,
                    data,
                )
                probe = asarray_probe_split(
                    operator,
                    num_gpu,
                    num_tile,
                    probe,
                )
                print(len(probe), probe[0].shape)
                print(scan[0].shape, scan[1].shape, scan[2].shape, scan[3].shape, scan[5].shape)
                result = {
                    'psi': pool.bcast(psi.astype('complex64')),
                    'probe': probe,
                    'scan': scan,
                }
                for key, value in kwargs.items():
                    if np.ndim(value) > 0:
                        kwargs[key] = pool.bcast(value)

            cost = 0
            for i in range(num_iter):
                result['probe'] = _rescale_obj_probe(operator, pool, num_gpu, num_tile,
                                                     data, result['psi'],
                                                     result['scan'],
                                                     result['probe'])
                kwargs.update(result)
                result = getattr(solvers, algorithm)(
                    operator,
                    pool,
                    num_gpu=num_gpu,
                    num_tile=num_tile,
                    data=data,
                    **kwargs,
                )
                # Check for early termination
                if i > 0 and abs((result['cost'] - cost) / cost) < rtol:
                    logger.info(
                        "Cost function rtol < %g reached at %d "
                        "iterations.", rtol, i)
                    break
                cost = result['cost']

            if (num_gpu > 1):
                result['scan'] = pool.gather(result['scan'], axis=1)
                for k, v in result.items():
                    if isinstance(v, list):
                        result[k] = v[0]
        return {k: operator.asnumpy(v) for k, v in result.items()}
    else:
        raise ValueError(
            "The '{}' algorithm is not an available.".format(algorithm))


def _rescale_obj_probe(operator, pool, num_gpu, num_tile, data, psi, scan, probe):
    """Keep the object amplitude around 1 by scaling probe by a constant."""
    # TODO: add multi-GPU support
    def f(mat):
        return np.sum(np.ravel(mat))

    #if (num_gpu > 1):
    #    scan = pool.gather(scan, axis=1)
    #    data = pool.gather(data, axis=1)
    #    psi = psi[0]
    #    probe = probe[0]

    #intensity = operator._compute_intensity(data, psi, scan, probe)
    intensity = list(pool.map(operator._compute_intensity, data, psi, scan, probe))

    data_norm = list(pool.map(f, data))
    inte_norm = list(pool.map(f, intensity))
    for i in range(num_gpu):
        data_norm[i] = np.expand_dims(data_norm[i], axis=0)
        inte_norm[i] = np.expand_dims(inte_norm[i], axis=0)
    data_norm = pool.gather(data_norm, axis=0)
    inte_norm = pool.gather(inte_norm, axis=0)
    data_norm = np.sqrt(np.sum(data_norm[:num_tile]))
    inte_norm = np.sqrt(np.sum(inte_norm))
    print('res=', data_norm, inte_norm)
    rescale = data_norm / inte_norm
    print('rescale=', type(rescale), rescale.shape, rescale)
    exit()
    #rescale = (np.linalg.norm(np.ravel(np.sqrt(data))) /
    #           np.linalg.norm(np.ravel(np.sqrt(intensity))))

    logger.info("object and probe rescaled by %f", rescale)

    probe *= rescale

    if (num_gpu > 1):
        probe = pool.bcast(probe)
        del scan
        del data

    return probe


def asarray_multi_split(op, gpu_count, num_tile, scan_cpu, data_cpu, *args, **kwargs):
    """Split scan and data and distribute to multiple GPUs.

    Instead of spliting the arrays based on the scanning order, we split
    them in accordance with the scan positions corresponding to the object
    sub-images. For example, if we divide a square object image into four
    sub-images, then the scan positions on the top-left sub-image and their
    corresponding diffraction patterns will be grouped into the first chunk
    of scan and data.

    """
    scanmlist = [None] * gpu_count
    datamlist = [None] * gpu_count
    nscan = scan_cpu.shape[1]
    tmplist = [0] * nscan
    counter = [0] * num_tile
    xmax = np.amax(scan_cpu[:, :, 0])
    ymax = np.amax(scan_cpu[:, :, 1])
    for e in range(nscan):
        xgpuid = scan_cpu[0, e, 0] // (xmax / (num_tile // 2)) - int(
            scan_cpu[0, e, 0] != 0 and scan_cpu[0, e, 0] %
            (xmax / (num_tile // 2)) == 0)
        ygpuid = scan_cpu[0, e, 1] // (ymax / 2) - int(
            scan_cpu[0, e, 1] != 0 and scan_cpu[0, e, 1] % (ymax / 2) == 0)
        idx = int(xgpuid * 2 + ygpuid)
        tmplist[e] = idx
        counter[idx] += 1
    for i in range(num_tile):
        tmpscan = np.zeros(
            [scan_cpu.shape[0], counter[i], scan_cpu.shape[2]],
            dtype=scan_cpu.dtype,
        )
        tmpdata = np.zeros(
            [
                data_cpu.shape[0], counter[i], data_cpu.shape[2],
                data_cpu.shape[3]
            ],
            dtype=data_cpu.dtype,
        )
        c = 0
        for e in range(nscan):
            if tmplist[e] == i:
                tmpscan[:, c, :] = scan_cpu[:, e, :]
                tmpdata[:, c] = data_cpu[:, e]
                c += 1
        for p in range(gpu_count//num_tile):
            scanmlist[p*num_tile+i] = op.asarray(tmpscan, device=(p*num_tile+i))
            datamlist[p*num_tile+i] = op.asarray(tmpdata, device=(p*num_tile+i))
        del tmpscan
        del tmpdata

    return scanmlist, datamlist

def asarray_probe_split(op, gpu_count, num_tile, probe_cpu, *args, **kwargs):
    """Split probes and distribute them to multiple GPUs.

    The probes might be replicated multiple times (depending on the number of tiles).
    The destination GPU ids of the distribution have a stride. For example, if the
    probes are divided to two groups and the image is divided to two tiles, then GPU0
    will hold Probe1 & Image1, GPU1 will hold Probe1 & Image2, GPU2 will hold
    Probe2 & Image1, and GPU3 will hold Probe2 & Image2.

    """
    probelist = [None] * gpu_count
    block_size = probe_cpu.shape[-3] // (gpu_count // num_tile)
    print(block_size)
    for i in range(num_tile):
        for j in range(gpu_count//num_tile):
            probelist[j*num_tile+i] = op.asarray(
                    probe_cpu[:, :, :, block_size*j:block_size*(j+1)],
                    device=(j*num_tile+i),
            )

    return probelist
