import logging

from tike.opt import conjugate_gradient, line_search
from ..position import update_positions_pd
import cupy as cp
import concurrent.futures as cf

logger = logging.getLogger(__name__)


def combined(
    op,
    num_gpu, data, probe, scan, psi,
    recover_psi=True, recover_probe=True, recover_positions=False,
    cg_iter=32,
    **kwargs
):  # yapf: disable
    """Solve the ptychography problem using a combined approach.

    """
    if recover_psi:
        op_list = [None] * num_gpu
        sendbuf = [None] * num_gpu
        recvbuf = [None] * num_gpu
        for i in range(num_gpu):
            op_list[i] = op
        gpu_list = range(num_gpu)
        for i in range(cg_iter):
            with cf.ThreadPoolExecutor(max_workers=num_gpu) as executor:
                psi_out = executor.map(
                    update_object,
                    op_list,
                    gpu_list,
                    data,
                    psi,
                    scan,
                    probe,
                )
            psi0 = psi
            psi = list(psi_out)

            # --------p2p comm-------
            #pw = probe[0].shape[4]
            #px = (psi[0].shape[1] - pw) // (num_gpu // 2)
            #py = (psi[0].shape[2] - pw) // 2
            #rx = (psi[0].shape[1] - pw) % (num_gpu // 2)
            #ry = (psi[0].shape[2] - pw) % 2
            #print('px', i, px, py, pw, rx, ry, (px*(1//2)+px-1+rx+pw))
            #for g in range(num_gpu):
            #    idx = g // 2
            #    idy = g % 2
            #    with cp.cuda.Device(g):
            #        sendbuf[g] = (psi[g][:, px*idx:(px*idx+px-1+rx+pw), py*idy:(py*idy+py-1+ry+pw)] -
            #            psi0[g][:, px*idx:(px*idx+px-1+rx+pw), py*idy:(py*idy+py-1+ry+pw)])
            #count = 0
            ##print('psi', i, type(sendbuf[0]), sendbuf[0].dtype, sendbuf[0].shape, sendbuf[0][:, :, 4])
            #while count < (num_gpu - 2):
            #    for s in range(count, count+4):
            #        for r in range(count, count+4):
            #            if r != s:
            #                with cp.cuda.Device(s):
            #                    cpu_tmp = cp.asnumpy(sendbuf[s])
            #                with cp.cuda.Device(r):
            #                    idx = r // 2
            #                    idy = r % 2
            #                    recvbuf[r] = cp.asarray(cpu_tmp)
            #                    psi[r][:, px*idx:(px*idx+px-1+rx+pw), py*idy:(py*idy+py-1+ry+pw)] += recvbuf[r]
            #                    #print('psi', i, type(recvbuf[r]), recvbuf[r].dtype, recvbuf[r].shape, recvbuf[r][:, :, 4])
            #    count += 2
            #else:
            #    if count == 0:
            #        for s in range(count, count+2):
            #            for r in range(count, count+2):
            #                if r != s:
            #                    with cp.cuda.Device(s):
            #                        cpu_tmp = cp.asnumpy(sendbuf[s])
            #                    with cp.cuda.Device(r):
            #                        idx = r // 2
            #                        idy = r % 2
            #                        recvbuf[r] = cp.asarray(cpu_tmp)
            #                        psi[r][:, px*idx:(px*idx+px-1+rx+pw), py*idy:(py*idy+py-1+ry+pw)] += recvbuf[r]
            #                        print('psi', i, type(recvbuf[r]), recvbuf[r].dtype, recvbuf[r].shape, recvbuf[r][:, :, 4])
            #        print('else', count)
            ##print('psi', i, type(sendbuf[1]), sendbuf[1].dtype, sendbuf[1].shape, sendbuf[1][:, :, 4])

            # --------all reduce-------
            comms = op.nccl_init(num_gpu, list(gpu_list))
            for g in range(num_gpu):
                with cp.cuda.Device(g):
                    sendbuf[g] = (psi[g] - psi0[g])
            op.nccl_comm(comms, 'allReduce', sendbuf, sendbuf)
            for g in range(num_gpu):
                with cp.cuda.Device(g):
                    psi[g] = (sendbuf[g] + psi0[g])
        exit()
        #psi, cost = update_object(
        #    op,
        #    num_gpu,
        #    data,
        #    psi,
        #    scan,
        #    probe,
        #    num_iter=cg_iter,
        #)

    if recover_probe:
        probe, cost = update_probe(
            op,
            num_gpu,
            data,
            psi,
            scan,
            probe,
            num_iter=cg_iter,
        )

    if recover_positions:
        scan, cost = update_positions_pd(op, data, psi, probe, scan)

    return {'psi': psi, 'probe': probe, 'cost': cost, 'scan': scan}


def update_probe(op, num_gpu, data, psi, scan, probe, num_iter=1):
    """Solve the probe recovery problem."""
    # TODO: add multi-GPU support
    if (num_gpu > 1):
        scan = op.asarray_multi_fuse(num_gpu, scan)
        data = op.asarray_multi_fuse(num_gpu, data)
        psi = psi[0]
        probe = probe[0]

    # TODO: Cache object patche between mode updates
    for m in range(probe.shape[-3]):

        def cost_function(mode):
            return op.cost(data, psi, scan, probe, m, mode)

        def grad(mode):
            # Use the average gradient for all probe positions
            return op.xp.mean(
                op.grad_probe(data, psi, scan, probe, m, mode),
                axis=(1, 2),
                keepdims=True,
            )

        probe[..., m:m + 1, :, :], cost = conjugate_gradient(
            op.xp,
            x=probe[..., m:m + 1, :, :],
            cost_function=cost_function,
            grad=grad,
            num_iter=num_iter,
        )

    if (num_gpu > 1):
        probe = op.asarray_multi(num_gpu, probe)
        del scan
        del data

    logger.info('%10s cost is %+12.5e', 'probe', cost)
    return probe, cost


def update_object(op, gpu_id, data, psi, scan, probe, num_gpu=1, num_iter=1):
    """Solve the object recovery problem."""

    def cost_function(psi):
        return op.cost(data, psi, scan, probe)

    def grad(psi):
        return op.grad(data, psi, scan, probe)

    def cost_function_multi(psi, **kwargs):
        return op.cost_multi(num_gpu, data, psi, scan, probe, **kwargs)

    def grad_multi(psi):
        return op.grad_multi(num_gpu, data, psi, scan, probe)

    def dir_multi(*args):
        return op.dir_multi(num_gpu, *args)

    def update_multi(psi, *args):
        return op.update_multi(num_gpu, psi, *args)

    with cp.cuda.Device(gpu_id):
        if (num_gpu <= 1):
            psi, cost = conjugate_gradient(
                op.xp,
                x=psi,
                cost_function=cost_function,
                grad=grad,
                num_gpu=num_gpu,
                num_iter=num_iter,
            )
        else:
            psi, cost = conjugate_gradient(
                op.xp,
                x=psi,
                cost_function=cost_function_multi,
                grad=grad_multi,
                dir_multi=dir_multi,
                update_multi=update_multi,
                num_gpu=num_gpu,
                num_iter=num_iter,
            )

    logger.info('%10s cost is %+12.5e', 'object', cost)
    #return psi, cost
    return psi
