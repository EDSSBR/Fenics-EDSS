"""*DLOAD de pressao (Pn) nas faces dos elementos. Item 6 do plano: ainda nao implementado."""

from inp_capabilities import InpNotImplementedError


def apply_pressure_loads(ctx, b, model, step):
    raise InpNotImplementedError(
        "*DLOAD (pressao em face) ainda nao implementado no FenicsEDSS; "
        "o job e recusado em vez de usar aproximacao."
    )
