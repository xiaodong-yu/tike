"""Define a MPI wrapper for inter-node communications.."""

__author__ = "Xiaodong Yu"
__copyright__ = "Copyright (c) 2021, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['MPIComm']

from mpi4py import MPI
import warnings

import cupy as cp
import numpy as np


class MPIComm:
    """A class for python MPI wrapper.

    Many clusters do not support inter-node GPU-GPU
    communications, so we first gather the data into
    main memory then communicate them.

    Attributes
    ----------
    gpu_count : int
        The number of GPUs on each node.

    """

    def __init__(self, gpu_count):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.gpu_count = gpu_count
        self.xp = cp

    def get_rank():
        return self.rank

    def get_size():
        return self.size

    def p2p(self, sendbuf, src=0, dest=1, tg=0, **kwargs):
        """Send data from a source to a designated destination."""

        if sendbuf is None:
            raise ValueError(f"Sendbuf can't be empty.")
        if self.rank == src:
            self.comm.Send(sendbuf, dest=dest, tag=tg, **kwargs)
        elif self.rank == dest:
            info = MPI.Status()
            recvbuf = np.empty(sendbuf.shape, sendbuf.dtype)
            self.comm.Recv(recvbuf,
                    source=src, tag=tg, status=info, **kwargs)
            return recvbuf

    def Bcast(self, data, int root=0):
        """Send data from a root to all processes."""

        if data is None:
            raise ValueError(f"Broadcast data can't be empty.")
        if self.rank == root:
            data = data
        else:
            data = np.empty(data.shape, data.dtype)
        self.comm.Bcast(data, root)
        return data


        return list(self.map(f, self.workers))

    def Gather(self, sendbuf, int dest=0):
        """Take data from all processes into one destination."""

        if sendbuf is None:
            raise ValueError(f"Gather data can't be empty.")
        if self.rank == dest:
            recvbuf = np.empty(sendbuf.shape, sendbuf.dtype)
        self.comm.Scatter(sendbuf, recvbuf, dest)
        if self.rank == dest:
            return recvbuf

    def Scatter(self, sendbuf, int src=0):
        """Spread data from a source to all processes."""

        if sendbuf is None:
            raise ValueError(f"Scatter data can't be empty.")
        recvbuf = np.empty(sendbuf.shape, sendbuf.dtype)
        self.comm.Scatter(sendbuf, recvbuf, src)
        return recvbuf

    def Allreduce(self, x: list, **kwargs):
        """Combines data from all processes and distributes
        the result back to all processes."""

        sendbuf = np.zeros(x[0].shape, x[0].dtype)
        for i in range(self.gpu_count):
            with cp.cuda.Device(i):
                sendbuf += cp.asnumpy(x[i])
        recvbuf = np.empty(sendbuf.shape, sendbuf.dtype)
        self.comm.Allreduce(sendbuf, recvbuf, op=MPI.SUM)
        data = cp.asarray(recvbuf)

        return data
