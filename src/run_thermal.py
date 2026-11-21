#!/usr/bin/env python3
"""
Executa condução térmica a partir de um arquivo .inp.
"""

import os
import sys
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import ufl
import basix.ufl

sys.path.insert(0, "/app/src")
from inp_parser import parse_inp, summary
from job_io import write_vtk_function


def run_thermal(inp_path: str, output_base: str = "/app/output/thermal", debug: int = 0):
    model = parse_inp(inp_path, debug=debug)
    if debug:
        print(summary(model))
        print("---")

    if model.analysis_type != "heat":
        print("Aviso: análise não é térmica")

    # Monta malha (reusa a lógica do run_from_inp)
    from run_from_inp import build_mesh_from_inp
    domain, node_map = build_mesh_from_inp(model, debug=debug)
    V = fem.functionspace(domain, ("Lagrange", 1))

    # Material (pega o primeiro com k)
    mat = None
    for m in model.materials.values():
        if m.k > 0:
            mat = m
            break
    if mat is None:
        mat = next(iter(model.materials.values()))
    k = default_scalar_type(mat.k if mat.k > 0 else 1.0)
    if debug:
        print(f"Condutividade: {float(k)}")

    # BCs de temperatura
    bcs = []
    coords = domain.geometry.x
    for bc in model.boundary:
        if bc.dof != 11:  # DOF 11 = temperatura no CalculiX
            continue
        if bc.node not in model.nodes:
            continue
        n = model.nodes[bc.node]
        def marker(x, nx=n.x, ny=n.y, nz=n.z):
            return (np.isclose(x[0], nx, atol=1e-6) &
                    np.isclose(x[1], ny, atol=1e-6) &
                    np.isclose(x[2], nz, atol=1e-6))
        dofs = fem.locate_dofs_geometrical(V, marker)
        if len(dofs) > 0:
            bc_val = fem.Constant(domain, default_scalar_type(bc.value))
            bcs.append(fem.dirichletbc(bc_val, dofs, V))

    if debug:
        print(f"BCs de temperatura: {len(bcs)}")

    # Formulação
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    a = k * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = fem.Constant(domain, default_scalar_type(0.0)) * v * ufl.dx

    problem = LinearProblem(
        a, L, bcs=bcs,
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="thermal_",
    )
    uh = problem.solve()
    uh.name = "temperature"

    # Saída somente no diretório de output_base (não em /app/output global).
    out_dir = os.path.dirname(os.path.abspath(output_base))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with io.XDMFFile(domain.comm, f"{output_base}.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(uh)
    write_vtk_function(domain.comm, uh, output_base, debug=debug)

    tmax = uh.x.array.max()
    tmin = uh.x.array.min()
    if debug:
        print(f"Temperatura: min={tmin:.2f}, max={tmax:.2f}")
    return uh


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 run_thermal.py arquivo.inp")
        sys.exit(1)
    run_thermal(sys.argv[1])


def run_heat_steady(model, step, output_base: str, debug: int = 0, solver: str = "lu",
                    rtol: float = 1e-10, auto_lu_max_dofs=None):
    """Conducao estacionaria (*HEAT TRANSFER, STEADY STATE) de um passo efetivo.

    Condutividade por celula via *SOLID SECTION (sem material padrao); temperatura
    prescrita no dof 11 do *BOUNDARY. Sem fluxo/conveccao (CFLUX/DFLUX/FILM sao
    recusados antes, em inp_capabilities). Saida: NT nodal (``temperature``).
    """
    import time as _time
    from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
    from petsc4py import PETSc
    from fc_assembly import CoordLookup, inp_element_to_cell, inp_node_to_geom
    from fc_profile import FCProfileError
    from fc_solver import solve_linear
    from inp_parser import materials_by_element_strict
    from job_io import normalize_rtol, normalize_solver
    from run_from_inp import build_mesh_from_inp, element_degree, element_dx, resolve_solver_choice

    solver = normalize_solver(solver)
    rtol = normalize_rtol(rtol, solver)
    if not step.heat_steady:
        raise FCProfileError("*HEAT TRANSFER transiente nao implementado (use STEADY STATE).")
    domain, _node_map = build_mesh_from_inp(model, debug=debug)
    domain.topology.create_connectivity(0, domain.topology.dim)
    domain.topology.create_connectivity(domain.topology.dim, 0)
    nid_to_geom = inp_node_to_geom(domain, model)
    eid_to_cell = inp_element_to_cell(domain, model, nid_to_geom)
    degree = element_degree(model)
    V = fem.functionspace(domain, ("Lagrange", degree))

    mats = materials_by_element_strict(model, "conductivity")
    Q = fem.functionspace(domain, ("DG", 0))
    k_func = fem.Function(Q)
    for eid, mat in mats.items():
        k_func.x.array[eid_to_cell[eid]] = mat.k
    k_func.x.scatter_forward()

    # Temperatura prescrita: dof 11. Outro dof num passo termico e erro.
    coords = np.asarray(V.tabulate_dof_coordinates(), dtype=np.float64).reshape(-1, 3)
    lookup = CoordLookup(coords)
    values = {}
    for bc in step.boundary:
        if bc.dof != 11:
            raise FCProfileError(
                "Passo termico com *BOUNDARY no dof {0} (no {1}); so o dof 11 (temperatura) "
                "e implementado na termica.".format(bc.dof, bc.node))
        node = model.nodes.get(bc.node)
        if node is None:
            raise FCProfileError("BOUNDARY referencia no inexistente: {0}".format(bc.node))
        idx, status, mind = lookup.find((node.x, node.y, node.z))
        if status != "ok":
            raise FCProfileError("BOUNDARY no {0}: dof termico {1} (dist={2}).".format(bc.node, status, mind))
        values[int(idx)] = float(bc.value)
    if not values:
        raise RuntimeError("Passo termico sem temperatura prescrita (*BOUNDARY dof 11): problema singular.")
    dofs = np.array(sorted(values), dtype=np.int32)
    T_bc = fem.Function(V)
    T_bc.x.array[dofs] = np.array([values[int(d)] for d in dofs])
    bcs = [fem.dirichletbc(T_bc, dofs)]

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx = element_dx(domain, next(iter(model.elements.values())).type)
    a_form = fem.form(k_func * ufl.inner(ufl.grad(u), ufl.grad(v)) * dx)
    L_form = fem.form(fem.Constant(domain, default_scalar_type(0.0)) * v * dx)
    A = assemble_matrix(a_form, bcs=bcs)
    A.assemble()
    b = assemble_vector(L_form)
    apply_lifting(b, [a_form], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)
    ndofs = int(V.dofmap.index_map.size_global)
    solver_used, auto_info = resolve_solver_choice(solver, ndofs, auto_lu_max_dofs)
    uh = fem.Function(V)
    info = solve_linear(A, b, uh, V, solver=solver_used, rtol=rtol, debug=debug, near_nullspace="none")
    if auto_info:
        info["auto"] = auto_info
    uh.name = "temperature"
    write_vtk_function(domain.comm, uh, output_base, debug=debug)
    tmin = float(uh.x.array.min())
    tmax = float(uh.x.array.max())
    if debug:
        print(f"Temperatura: min={tmin:.4f}, max={tmax:.4f}")
    extra = {
        "honored": ["NT"],
        "artifacts": [{"path": output_base + ".pvd", "field": "NT", "location": "nodal-CG{0}".format(degree),
                       "space": "Lagrange-P{0}".format(degree)}],
        "reactions": None,
        "solver_info": info,
        "analysis": "heat",
        "t_min": tmin,
        "t_max": tmax,
    }
    return uh, extra
