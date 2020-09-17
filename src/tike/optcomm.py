"""Defines a class for fine-grained multi-GPU communications."""

__author__ = "Xiaodong Yu"
__copyright__ = "Copyright (c) 2020, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['OptComm']

from concurrent.futures import ThreadPoolExecutor
import os
import warnings

import numpy as np
import cupy as cp


class OptComm(ThreadPoolExecutor):

    def __init__(self, num_gpu: int, pair_list: list):
        super().__init__(num_gpu)
        self.num_gpu = num_gpu
        self.gpu_list = []
        if pair_list is None:
            self.gpu_list = list(range(num_gpu))
        else:
            self.num_pairs = len(pair_list)
            self.pair_list = pair_list
            for pair in self.pair_list:
                self.gpu_list.append(pair[0])
                self.gpu_list.append(pair[1])
                if cp.cuda.runtime.deviceCanAccessPeer(pair[0], pair[1]) == True:
                    print('pair', pair[0], pair[1], cp.cuda.runtime.deviceCanAccessPeer(pair[0], pair[1]))
                    with cp.cuda.Device(pair[0]):
                        cp.cuda.runtime.deviceEnablePeerAccess(pair[1])
                    with cp.cuda.Device(pair[1]):
                        cp.cuda.runtime.deviceEnablePeerAccess(pair[0])
        self.xp = cp

    def create_bufs(self, subpsi_x, subpsi_y, probe_size):
        send_buf = []
        recv_buf = []
        comb_buf = []
        lmar_buf = []
        rmar_buf = []
        for i in self.gpu_list:
            print('i', i)
            with cp.cuda.Device(i):
                send_buf.append(cp.zeros([subpsi_x, subpsi_y], dtype='complex64')+i)
                recv_buf.append(cp.zeros([subpsi_x, subpsi_y], dtype='complex64'))
                comb_buf.append(cp.zeros([(subpsi_x * 2 - probe_size), subpsi_y], dtype='complex64', order='F'))
                lmar_buf.append(cp.zeros([(subpsi_x * 2 - probe_size), probe_size], dtype='complex64', order='F'))
                rmar_buf.append(cp.zeros([(subpsi_x * 2 - probe_size), probe_size], dtype='complex64', order='F'))
        return {'send_buf': send_buf, 'recv_buf': recv_buf, 'comb_buf': comb_buf, 'lmar_buf': lmar_buf, 'rmar_buf': rmar_buf}

    def _peerasync_comm(self, dst_buf, dst_id: int, src_buf, src_id: int):
        print ('dst', dst_id, src_id)
        with cp.cuda.Stream() as stream_dtod:
            print("p2p:", cp.cuda.runtime.deviceCanAccessPeer(dst_id, src_id))
            cp.cuda.runtime.memcpyPeerAsync(dst_buf.data.ptr, dst_id, src_buf.data.ptr, src_id, dst_buf.size * dst_buf.itemsize, stream_dtod.ptr)
            stream_dtod.synchronize()

    def nvl_exchange(self, src_ids, dst_ids, data, buffers):

        (subpsi_x, subpsi_y) = buffers['send_buf'][0].shape
        probe_size = buffers['lmar_buf'][0].shape[1]
        print('type', subpsi_x, subpsi_y, probe_size)

        def f(src, dst, pos):
            with cp.cuda.Device(src):
                send_buf = data[pos][...]
                with cp.cuda.Device(dst):
                    recv_buf = cp.zeros(send_buf.shape, dtype='complex64')
                self._peerasync_comm(recv_buf, dst, send_buf, src)
                cp.cuda.runtime.deviceSynchronize()
            return (send_buf, recv_buf)

        output = super().map(f, src_ids, dst_ids, list(range(self.num_gpu)))
        buf_output = list(output)
        send_bufs = []
        recv_bufs = []
        for i in range(len(buf_output)//2):
            send_bufs.append(buf_output[i*2][0])
            send_bufs.append(buf_output[i*2+1][0])
            recv_bufs.append(buf_output[i*2+1][1])
            recv_bufs.append(buf_output[i*2][1])
        #buffers['recv_buf'] = []
        #for l in recv_bufs:
        #    buffers['recv_buf'].append(l)

        def comb(src, pos, send_buf, recv_buf):
            with cp.cuda.Device(src):
                comb_buf = cp.zeros([(subpsi_x * 2 - probe_size), subpsi_y], dtype='complex64', order='F')
                if (pos%2) is 0:
                    comb_buf[:subpsi_x, ...] = send_buf
                    comb_buf[(subpsi_x-probe_size):, ...] += recv_buf
                else:
                    comb_buf[:subpsi_x, ...] = recv_buf
                    comb_buf[(subpsi_x-probe_size):, ...] += send_buf
            return comb_buf

        output = super().map(comb, src_ids, list(range(self.num_gpu)), send_bufs, recv_bufs)
        buffers['comb_buf'] = list(output)

        #with cp.cuda.Device(2) as src_device:
        #    buffers['recv_buf'][0] += buffers['recv_buf'][0]
        #print('test2_buf', buffers['comb_buf'][0])
        #print('send_buf', buffers['send_buf'][2])
        #print('send_buf', buffers['recv_buf'][0])

    def pci_ring(self, gpu_ids, positions, data, buffers):

        (subpsi_x, subpsi_y) = buffers['send_buf'][0].shape
        probe_size = buffers['lmar_buf'][0].shape[1]

        def f(gpu_id, pos):
            with cp.cuda.Device(gpu_id[0]):
                if (pos[0]%2) is 0:
                    send_buf = data[pos[0]][:, (subpsi_y - probe_size):]
                else:
                    send_buf = data[pos[0]][:, :probe_size]
                with cp.cuda.Device(gpu_id[1]):
                    recv_buf = cp.zeros([(subpsi_x * 2 - probe_size), probe_size], dtype='complex64', order='F')
                    self._peerasync_comm(recv_buf, gpu_id[1], send_buf, gpu_id[0])
                    if (pos[1]%2) is 0:
                        data[pos[1]][:, :probe_size] += recv_buf
                    else:
                        data[pos[1]][:, (subpsi_y - probe_size):] += recv_buf
                cp.cuda.runtime.deviceSynchronize()
                return data[pos[1]]

        output = super().map(f, gpu_ids, positions)
        counter = 0
        for buff in output:
            buffers['comb_buf'][positions[counter][1]] = buff
            counter += 1

    def nvl_ring(self, gpu_ids, positions, data, buffers):

        (subpsi_x, subpsi_y) = buffers['send_buf'][0].shape
        probe_size = buffers['lmar_buf'][0].shape[1]

        def f(gpu_id, pos):
            with cp.cuda.Device(gpu_id[0]):
                if (pos[0]%2) is 0:
                    send_buf = data[pos[0]][:, :probe_size]
                else:
                    send_buf = data[pos[0]][:, (subpsi_y - probe_size):]
                with cp.cuda.Device(gpu_id[1]):
                    recv_buf = cp.zeros([(subpsi_x * 2 - probe_size), probe_size], dtype='complex64', order='F')
                    self._peerasync_comm(recv_buf, gpu_id[1], send_buf, gpu_id[0])
                    if (pos[1]%2) is 0:
                        data[pos[1]][:, (subpsi_y - probe_size):] = recv_buf
                    else:
                        data[pos[1]][:, :probe_size] = recv_buf
                cp.cuda.runtime.deviceSynchronize()
                return data[pos[1]]

        output = super().map(f, gpu_ids, positions)
        counter = 0
        for buff in output:
            buffers['comb_buf'][positions[counter][1]] = buff
            counter += 1

    def async_exec(self, func, src_ids, dst_ids, data, buffers, *iterables, **kwargs):

        (subpsi_x, subpsi_y) = buffers['send_buf'][0].shape
        probe_size = buffers['lmar_buf'][0].shape[1]

        gpu_lists = [self.gpu_list] * self.num_gpu
        #print('pair_list', gpu_lists)
        def f(pos, gpu_list, *args):
            with cp.cuda.Device(gpu_list[pos]):
                buffers['comb_buf'][pos] = cp.zeros([(subpsi_x * 2 - probe_size), subpsi_y], dtype='complex64', order='F')
                with cp.cuda.Stream() as stream_exec:
                    output = func(*args)
                    with cp.cuda.Stream() as stream_comm:
                        idx = pos % 2
                        idy = pos // 2
                        spx = subpsi_x - probe_size
                        spy = subpsi_y - probe_size
                        buffers['send_buf'][pos][...] = data[pos][:, spx*idx:(spx*(idx+1)+probe_size), spy*idy:(spy*(idy+1)+probe_size)].reshape(subpsi_x, subpsi_y)

                        if (pos%2) is 0:
                            cp.cuda.runtime.memcpyPeerAsync(buffers['recv_buf'][pos].data.ptr, gpu_list[pos],
                                    buffers['send_buf'][pos+1].data.ptr, gpu_list[pos+1],
                                    buffers['recv_buf'][pos].size * buffers['recv_buf'][pos].itemsize,
                                    stream_comm.ptr)
                            stream_comm.synchronize()
                            buffers['comb_buf'][pos][:subpsi_x, ...] = buffers['send_buf'][pos]
                            buffers['comb_buf'][pos][(subpsi_x-probe_size):, ...] += buffers['recv_buf'][pos]
                            if pos is not 0:
                                #with cp.cuda.Device(gpu_list[pos-2]):
                                #    buffers['comb_buf'][pos-2][:subpsi_x, ...] = buffers['send_buf'][pos-2]
                                #    buffers['comb_buf'][pos-2][(subpsi_x-probe_size):, ...] += buffers['recv_buf'][pos-2]
                                cp.cuda.runtime.memcpyPeerAsync(buffers['lmar_buf'][pos].data.ptr, gpu_list[pos],
                                        buffers['comb_buf'][pos-2][:, (subpsi_y - probe_size):].data.ptr, gpu_list[pos-2],
                                        buffers['lmar_buf'][pos].size * buffers['lmar_buf'][pos].itemsize,
                                        stream_comm.ptr)
                                stream_comm.synchronize()
                                buffers['comb_buf'][pos][:, :probe_size] += buffers['lmar_buf'][pos]
                                #if pos is 4:
                                #    print(buffers['comb_buf'][pos])
                            if pos is not (len(gpu_list)-2):
                                cp.cuda.runtime.memcpyPeerAsync(buffers['rmar_buf'][pos].data.ptr, gpu_list[pos],
                                        buffers['comb_buf'][pos+1][:, (subpsi_y - probe_size):].data.ptr, gpu_list[pos+1],
                                        buffers['rmar_buf'][pos].size * buffers['rmar_buf'][pos].itemsize,
                                        stream_comm.ptr)
                                stream_comm.synchronize()
                                buffers['comb_buf'][pos][:, (subpsi_y - probe_size):] = buffers['rmar_buf'][pos]
                                #if pos is 4:
                                #    print(buffers['comb_buf'][pos])
                        else:
                            cp.cuda.runtime.memcpyPeerAsync(buffers['recv_buf'][pos].data.ptr, gpu_list[pos],
                                    buffers['send_buf'][pos-1].data.ptr, gpu_list[pos-1],
                                    buffers['recv_buf'][pos].size * buffers['recv_buf'][pos].itemsize,
                                    stream_comm.ptr)
                            stream_comm.synchronize()
                            buffers['comb_buf'][pos][:subpsi_x, ...] = buffers['recv_buf'][pos]
                            buffers['comb_buf'][pos][(subpsi_x-probe_size):, ...] += buffers['send_buf'][pos]
                            if pos is not (len(gpu_list)-1):
                                #cp.cuda.runtime.memcpyPeerAsync(buffers['comb_buf'][pos][:, (subpsi_y - probe_size):].data.ptr, gpu_list[pos],
                                #        buffers['comb_buf'][pos+2][:, :probe_size].data.ptr, gpu_list[pos+2],
                                #        buffers['comb_buf'][pos][:, (subpsi_y - probe_size):].size * buffers['comb_buf'][pos][:, (subpsi_y - probe_size):].itemsize,
                                #        stream_dtod.ptr)
                                cp.cuda.runtime.memcpyPeerAsync(buffers['rmar_buf'][pos].data.ptr, gpu_list[pos],
                                        buffers['comb_buf'][pos+2][:, :probe_size].data.ptr, gpu_list[pos+2],
                                        buffers['rmar_buf'][pos].size * buffers['rmar_buf'][pos].itemsize,
                                        stream_comm.ptr)
                                stream_comm.synchronize()
                                buffers['comb_buf'][pos][:, (subpsi_y - probe_size):] += buffers['rmar_buf'][pos]
                            if pos is not 1:
                                #with cp.cuda.Device(gpu_list[pos-2]):
                                #    buffers['comb_buf'][pos-2][:subpsi_x, ...] = buffers['send_buf'][pos-2]
                                #    buffers['comb_buf'][pos-2][(subpsi_x-probe_size):, ...] += buffers['recv_buf'][pos-2]
                                cp.cuda.runtime.memcpyPeerAsync(buffers['lmar_buf'][pos].data.ptr, gpu_list[pos],
                                        buffers['comb_buf'][pos-1][:, :probe_size].data.ptr, gpu_list[pos-1],
                                        buffers['lmar_buf'][pos].size * buffers['lmar_buf'][pos].itemsize,
                                        stream_comm.ptr)
                                stream_comm.synchronize()
                                buffers['comb_buf'][pos][:, :probe_size] = buffers['lmar_buf'][pos]
                                #if pos is 5:
                                #    #print(buffers['lmar_buf'][pos])
                                #    print(recv_buf)

                        stream_comm.synchronize()
                    stream_exec.synchronize()
                cp.cuda.runtime.deviceSynchronize()

                return output

        output = super().map(f, list(range(self.num_gpu)), gpu_lists, *iterables, **kwargs)

        return list(output)
