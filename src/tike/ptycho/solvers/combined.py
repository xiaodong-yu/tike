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
        for i in range(num_gpu):
            op_list[i] = op
        gpu_list = range(num_gpu)
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
        psi_list = list(psi_out)
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
