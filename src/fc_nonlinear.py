"""Estatica com NLGEOM (passo 7, 21/11/2026), seguindo o CalculiX 2.23.

Formulacao Lagrangiana total, Newton-Raphson com incrementos de carga do *STATIC.

Materiais por celula (via *SOLID SECTION):
* *HYPERELASTIC, NEO HOOKE (C10, D1), forma do CalculiX:
      W = C10 (I1_bar - 3) + (1/D1) (J - 1)^2,   I1_bar = J^(-2/3) tr(C)
* *ELASTIC com NLGEOM: St. Venant-Kirchhoff (S = lambda tr(E) I + 2 mu E, E de
  Green-Lagrange), como o CalculiX faz para *ELASTIC com NLGEOM.

Cargas *CLOAD sao forcas nodais de direcao fixa (nao seguidoras), como no CalculiX.
Cargas e deslocamentos prescritos variam linearmente no passo (rampa do valor do
passo anterior ao do passo atual). O estado final do passo e o equilibrio em t =
periodo; o controle de incremento so precisa chegar la. Tensao de saida: Cauchy
(sigma = P F^T / J), media simples nos pontos de integracao do CalculiX.
RF: forca interna nos DOFs prescritos (definicao do CalculiX); equilibrio de
momento na configuracao deformada.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, set_bc
from mpi4py import MPI

from fc_assembly import (
    apply_cloads,
    collect_constrained_dofs,
    component_dof_locator,
    require_serial_fc,
    residual_on_constrained,
    vec_as_array,
)
from fc_solver import solve_linear
from inp_parser import materials_by_element_strict
from job_io import normalize_rtol, normalize_solver, write_vtk_function

# Regra de integracao do CalculiX: C3D4 1 ponto, C3D8 2x2x2, C3D10 4 pontos.
NL_QUADRATURE_DEGREE = {"C3D4": 1, "C3D8": 2, "C3D10": 2}
# Tolerancias numericas do Newton (consistencia do proprio solver, nao criterio de aceite).
NEWTON_RTOL = 1e-10
NEWTON_MAX_IT = 20
CUTBACK = 0.25
GROWTH = 1.5


def nl_dx(ctx):
    deg = NL_QUADRATURE_DEGREE[ctx.etype]
    return ufl.dx(domain=ctx.domain, metadata={"quadrature_degree": deg})


class NonlinearForms:
    """Formas (residuo, jacobiano, tensoes) montadas uma vez por modelo."""

    def __init__(self, ctx, model):
        domain = ctx.domain
        V = ctx.V
        mats = materials_by_element_strict(model, "mechanical")
        Q = fem.functionspace(domain, ("DG", 0))
        lam, mu, c10, k1 = (fem.Function(Q) for _ in range(4))
        n = lam.x.array.size
        kinds = {}
        for eid, mat in mats.items():
            c = ctx.eid_to_cell[eid]
            if c >= n:
                raise RuntimeError("Indice de celula fora do espaco DG0 (serial esperado).")
            if mat.hyperelastic:
                c10.x.array[c] = mat.hyper_c10
                k1.x.array[c] = 1.0 / mat.hyper_d1
                kinds[mat.name] = {"law": "Neo-Hooke (CalculiX)", "C10": mat.hyper_c10, "D1": mat.hyper_d1}
            else:
                E, nu = mat.E, mat.nu
                lam.x.array[c] = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
                mu.x.array[c] = E / (2.0 * (1.0 + nu))
                kinds[mat.name] = {"law": "St. Venant-Kirchhoff (*ELASTIC com NLGEOM)", "E": E, "nu": nu}
        self.materials = kinds
        self.u = fem.Function(V)
        self.u.name = "displacement"
        v = ufl.TestFunction(V)
        du = ufl.TrialFunction(V)
        I = ufl.Identity(3)
        F = ufl.variable(I + ufl.grad(self.u))
        C = F.T * F
        J = ufl.det(F)
        Egl = 0.5 * (C - I)
        psi = (0.5 * lam * ufl.tr(Egl) ** 2 + mu * ufl.inner(Egl, Egl)
               + c10 * (J ** (-2.0 / 3.0) * ufl.tr(C) - 3.0) + k1 * (J - 1.0) ** 2)
        P = ufl.diff(psi, F)
        dx = nl_dx(ctx)
        res = ufl.inner(P, ufl.grad(v)) * dx
        self.res_form = fem.form(res)
        self.jac_form = fem.form(ufl.derivative(res, self.u, du))
        self.cauchy = (P * F.T) / J
        self.green = Egl
        self.ctx = ctx


def _load_vector(ctx, model, step):
    b = assemble_vector(ctx.L_compiled)
    applied = apply_cloads(ctx.V, b, model, step, True) if step is not None else {"Fx": 0.0, "Fy": 0.0, "Fz": 0.0}
    arr = vec_as_array(b)
    b.destroy()
    return arr, applied


def _increments(step):
    inc = list(step.static_increments or [])
    period = inc[1] if len(inc) > 1 and inc[1] > 0 else 1.0
    dt0 = inc[0] if len(inc) > 0 and inc[0] > 0 else period
    dtmin = inc[2] if len(inc) > 2 and inc[2] > 0 else min(dt0, 1e-5 * period)
    dtmax = inc[3] if len(inc) > 3 and inc[3] > 0 else period
    return min(dt0, period), period, dtmin, dtmax


def run_nlgeom_step(model, step, output_base, ctx, state, solver="lu", rtol=1e-10,
                    auto_lu_max_dofs=None, debug=0):
    """Um passo com NLGEOM. ``state`` guarda u e as cargas do passo anterior entre passos."""
    from run_from_inp import _cell_average_tensor, _check_fc_force_balance, resolve_solver_choice

    require_serial_fc(MPI.COMM_WORLD, True)
    solver = normalize_solver(solver)
    rtol = normalize_rtol(rtol, solver)
    if state.get("forms") is None:
        state["forms"] = NonlinearForms(ctx, model)
    nf = state["forms"]
    V, u = ctx.V, nf.u
    if state.get("u") is not None:
        u.x.array[:] = state["u"].x.array
    u.x.scatter_forward()
    solver_used, auto_info = resolve_solver_choice(solver, ctx.ndofs, auto_lu_max_dofs)

    boundary = step.boundary if step.boundary is not None else model.boundary
    bcs, rows = collect_constrained_dofs(V, model, True, boundary=boundary)
    if not rows:
        raise RuntimeError("Recusado: sistema sem DOFs prescritos (*BOUNDARY).")
    bc_dofs = np.array(sorted({d for _n, _c, d in rows}), dtype=np.int64)
    z = assemble_vector(ctx.L_compiled)
    z.set(0.0)
    set_bc(z, bcs)
    g1 = vec_as_array(z)[bc_dofs]
    z.destroy()
    g0 = u.x.array[bc_dofs].copy()

    f1, applied = _load_vector(ctx, model, step)
    f0, _ = _load_vector(ctx, model, state.get("loads_step"))

    dt, period, dtmin, dtmax = _increments(step)
    direct = bool(getattr(step, "static_direct", False))
    t = 0.0
    u_conv = u.x.array.copy()
    history: List[Dict[str, Any]] = []
    easy = 0
    last_info = None
    du = fem.Function(V)

    def newton(tau):
        nonlocal last_info
        u.x.array[bc_dofs] = g0 + tau * (g1 - g0)
        u.x.scatter_forward()
        f = f0 + tau * (f1 - f0)
        fnorm = float(np.max(np.abs(f))) if f.size else 0.0
        for it in range(NEWTON_MAX_IT + 1):
            rv = assemble_vector(nf.res_form)
            rint = vec_as_array(rv)
            rv.destroy()
            if not np.all(np.isfinite(rint)):
                return False, it, "residuo nao finito (J <= 0?)"
            r = rint - f
            r[bc_dofs] = 0.0
            scale = max(fnorm, float(np.max(np.abs(rint))), 1e-300)
            rmax = float(np.max(np.abs(r)))
            if debug:
                print("  Newton it={0} |R|max={1:.3e} escala={2:.3e}".format(it, rmax, scale), flush=True)
            if rmax <= NEWTON_RTOL * scale:
                return True, it, None
            if it == NEWTON_MAX_IT:
                return False, it, "Newton nao convergiu em {0} iteracoes (|R|max={1:.3e})".format(it, rmax)
            A = assemble_matrix(nf.jac_form, bcs=bcs)
            A.assemble()
            b = A.createVecLeft()
            b.array[:] = -r
            du.x.array[:] = 0.0
            try:
                last_info = solve_linear(A, b, du, V, solver=solver_used, rtol=rtol, debug=0)
            except Exception as exc:  # LU singular / KSP nao convergiu -> corta o incremento
                A.destroy(); b.destroy()
                return False, it, "solver linear falhou: {0}".format(exc)
            A.destroy(); b.destroy()
            u.x.array[:] += du.x.array
            u.x.scatter_forward()
        return False, NEWTON_MAX_IT, "Newton nao convergiu"

    while t < period * (1.0 - 1e-12):
        dt = min(dt, period - t)
        tau = (t + dt) / period
        ok, its, why = newton(tau)
        if ok:
            t += dt
            u_conv = u.x.array.copy()
            history.append({"time": t, "increment": dt, "newton_iterations": its})
            if debug:
                print("Incremento ok t={0:.6g} dt={1:.6g} its={2}".format(t, dt, its), flush=True)
            if not direct:
                easy = easy + 1 if its <= 4 else 0
                if easy >= 2:
                    dt = min(dt * GROWTH, dtmax)
        else:
            u.x.array[:] = u_conv
            u.x.scatter_forward()
            history.append({"time": t + dt, "increment": dt, "failed": why})
            if direct:
                raise RuntimeError("NLGEOM com *STATIC, DIRECT: incremento {0} falhou ({1}).".format(dt, why))
            dt *= CUTBACK
            easy = 0
            if dt < dtmin:
                raise RuntimeError(
                    "NLGEOM: incremento abaixo do minimo do *STATIC ({0:.3g} < {1:.3g}) em t={2:.6g}; "
                    "ultimo motivo: {3}. Nao e sucesso.".format(dt, dtmin, t, why))

    # ---- saida ----
    uh = fem.Function(V)
    uh.name = "displacement"
    uh.x.array[:] = u.x.array
    uh.x.scatter_forward()
    state["u"] = uh
    state["loads_step"] = step
    write_vtk_function(ctx.domain.comm, uh, output_base, debug=debug)

    from fc_fields import (interpolate_cell_scalar, interpolate_cell_tensor, interpolate_cell_vector,
                           principals_algebraic, tensor_array_from_function, write_functions_vtk)
    if ctx.etype == "C3D4":
        cauchy_f = interpolate_cell_tensor(ctx.domain, nf.cauchy, "cauchy_stress")
        strain_f = interpolate_cell_tensor(ctx.domain, nf.green, "strain")
        cell_location = "cell-DG0-constant-C3D4-P1"
    else:
        cauchy_f = _cell_average_tensor(ctx, nf.cauchy, "cauchy_stress")
        strain_f = _cell_average_tensor(ctx, nf.green, "strain")
        cell_location = "cell-DG0-average-of-integration-points-{0}".format(ctx.etype)
    sigma_cells = tensor_array_from_function(cauchy_f)
    principals, vm = principals_algebraic(sigma_cells)
    prin_f = interpolate_cell_vector(ctx.domain, principals, "principal_stress")
    vm_f = interpolate_cell_scalar(ctx.domain, vm, "von_mises")
    cell_pvd = write_functions_vtk(ctx.domain.comm, [cauchy_f, strain_f, prin_f, vm_f], output_base + "_cell")

    # Forca interna (RF = f_int nos prescritos) e equilibrio na configuracao deformada.
    rv = assemble_vector(nf.res_form)
    fext = assemble_vector(ctx.L_compiled)
    fext.array[:] = f1
    locator = component_dof_locator(V)

    def deformed(nid):
        node = model.nodes[nid]
        parent, _p, _v = locator.locate(node, 0)
        blk = int(parent[0]) // 3
        d = uh.x.array[3 * blk:3 * blk + 3]
        return (node.x + float(d[0]), node.y + float(d[1]), node.z + float(d[2]))

    coords = {nid: deformed(nid) for nid in {n for n, _c, _d in rows}}
    rx = residual_on_constrained(rv, fext, rows, coords)
    from fc_assembly import node_print_rf_totals
    np_totals = node_print_rf_totals(V, model, vec_as_array(rv), locator=locator)
    rf = fem.Function(V)
    rf.name = "reaction_force"
    rf.x.array[:] = rx["rf_constrained"][:rf.x.array.size]
    rf_pvd = write_functions_vtk(ctx.domain.comm, [rf], output_base + "_RF")

    totals, ext_c, ext_cm = rx["totals"], rx["ext_on_constrained"], rx["moment_ext_on_constrained_origin"]
    sum_rf = {k: totals["R" + k] + applied[k] - ext_c[k] for k in ("Fx", "Fy", "Fz")}
    mom_f = {"Mx": 0.0, "My": 0.0, "Mz": 0.0}
    force_char = moment_char = 0.0
    for cl in step.loads:
        if cl.node not in model.nodes:
            continue
        x, y, zc = deformed(cl.node)
        fx, fy, fz = [cl.value if cl.dof == k else 0.0 for k in (1, 2, 3)]
        m = (y * fz - zc * fy, zc * fx - x * fz, x * fy - y * fx)
        for key, val in zip(("Mx", "My", "Mz"), m):
            mom_f[key] += val
        force_char += abs(fx) + abs(fy) + abs(fz)
        moment_char += sum(abs(val) for val in m)
    for ent in rx["entries"]:
        x, y, zc = coords[ent["inp_node"]]
        val, comp = ent["R"], ent["component"]
        fx, fy, fz = [val if comp == k else 0.0 for k in (0, 1, 2)]
        force_char += abs(val)
        moment_char += abs(y * fz - zc * fy) + abs(zc * fx - x * fz) + abs(x * fy - y * fx)
    mom_sum = {k: rx["moment_R_origin"][k] + mom_f[k] - ext_cm[k] for k in ("Mx", "My", "Mz")}
    react_path = output_base + "_reactions.json"
    payload = {
        "totals": totals,
        "applied_totals": applied,
        "sum_R_plus_F": sum_rf,
        "loads_on_constrained_dofs": ext_c,
        "node_print_totals": np_totals,
        "rf_definition": "CalculiX: RF = forca interna nos DOFs prescritos (NLGEOM: configuracao deformada)",
        "moment_R_origin": rx["moment_R_origin"],
        "moment_F_origin": mom_f,
        "moment_R_plus_F_origin": mom_sum,
        "moment_configuration": "deformada (x = X + u)",
        "n_constrained_dofs": rx["n_constrained_dofs"],
        "residual_free_dofs_max_abs": rx["residual_free_dofs_max_abs"],
        "force_char": force_char,
        "moment_char": moment_char,
        "entries": rx["entries"],
    }
    with open(react_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    rv.destroy()
    fext.destroy()

    solver_info = dict(last_info or {})
    if auto_info:
        solver_info["auto"] = auto_info
    solver_info["nlgeom"] = {
        "method": "Newton-Raphson, Lagrangiano total",
        "period": period, "initial_increment": _increments(step)[0], "min_increment": dtmin,
        "max_increment": dtmax, "direct": direct,
        "newton_rtol": NEWTON_RTOL, "newton_max_it": NEWTON_MAX_IT,
        "increments": history,
        "materials": nf.materials,
    }
    extra = {
        "honored": ["U", "S", "E", "PRINCIPAL", "VON_MISES", "RF"],
        "artifacts": [
            {"path": output_base + ".pvd", "field": "U", "location": "nodal-CG{0}".format(ctx.degree),
             "space": "Lagrange-P{0}".format(ctx.degree)},
            {"path": cell_pvd, "field": "S,E,PRINCIPAL,VON_MISES", "location": cell_location,
             "stress": "Cauchy (NLGEOM)", "strain": "Green-Lagrange (NLGEOM)",
             "tensor_order": "row-major XX,XY,XZ,YX,YY,YZ,ZX,ZY,ZZ"},
            {"path": rf_pvd, "field": "RF", "location": "nodal-constrained-dofs-only"},
            {"path": react_path, "field": "RF_totals", "location": "json"},
        ],
        "reactions": payload,
        "applied": applied,
        "stress_cells": sigma_cells,
        "principals": principals,
        "von_mises": vm,
        "solver_info": solver_info,
        "analysis": "static-nlgeom",
        "u_max": float(np.max(np.linalg.norm(uh.x.array.reshape(-1, 3), axis=1))) if uh.x.array.size else 0.0,
    }
    _check_fc_force_balance(extra)
    return uh, extra
