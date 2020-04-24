import logging

from tike.opt import conjugate_gradient, line_search
from ..position import update_positions_pd

logger = logging.getLogger(__name__)


def combined(
    op,
    data, probe, scan, psi,
    recover_psi=True, recover_probe=True, recover_positions=False,
    cg_iter=4,
    **kwargs
):  # yapf: disable
    """Solve the ptychography problem using a combined approach.

    .. seealso:: tike.ptycho.divided
    """
    if recover_psi:
        psi, cost = update_object(op, data, psi, scan, probe, num_iter=cg_iter)

    if recover_probe:
        probe, cost = update_probe(op, data, psi, scan, probe, num_iter=cg_iter)

    if recover_positions:
        scan, cost = update_positions_pd(op, data, psi, probe, scan)

    return {'psi': psi, 'probe': probe, 'cost': cost, 'scan': scan}


def update_probe(op, data, psi, scan, probe, num_iter=1):
    """Solve the probe recovery problem."""

    def cost_function(probe):
        return op.cost(data, psi, scan, probe)

    for mode in range(probe.shape[-3]):

        def grad(probe):
            # Use the average gradient for all probe positions
            grad = op.xp.zeros_like(probe)
            grad[..., mode:mode+1, :, :] = op.xp.mean(
                op.grad_probe(data, psi, scan, probe, mode),
                axis=(1, 2),
                keepdims=True,
            )
            return grad

        probe, cost = conjugate_gradient(
            op.xp,
            x=probe,
            cost_function=cost_function,
            grad=grad,
            num_iter=num_iter,
        )

    logger.info('%10s cost is %+12.5e', 'probe', cost)
    return probe, cost


def update_object(op, data, psi, scan, probe, num_iter=1):
    """Solve the object recovery problem."""

    def cost_function(psi):
        return op.cost(data, psi, scan, probe)

    def grad(psi):
        return op.grad(data, psi, scan, probe)

    psi, cost = conjugate_gradient(
        op.xp,
        x=psi,
        cost_function=cost_function,
        grad=grad,
        num_iter=num_iter,
    )

    logger.info('%10s cost is %+12.5e', 'object', cost)
    return psi, cost
