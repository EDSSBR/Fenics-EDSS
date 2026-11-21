#!/usr/bin/env python3
"""
Executa a analise de um .inp do CalculiX (um caminho so, guiado pelo deck).

Estatica linear (C3D4, C3D8, C3D10; varios *STEP; *CLOAD; *DLOAD pressao) e
termica estacionaria. O que nao e implementado falha com erro explicito
(inp_capabilities.py). ``--profile fc`` so acrescenta as restricoes do perfil
FC; ``--restrictions`` aplica as restricoes do arquivo de configuracao.
"""

import json
import time
import sys
import os
import uuid
import numpy as np
from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
from petsc4py import PETSc
import ufl
import basix.ufl

sys.path.insert(0, "/app/src")
from inp_parser import (
    parse_inp, summary, Step, select_analysis, get_material_for_element,
    effective_steps, materials_by_element_strict,
)
from inp_capabilities import apply_restrictions, ignored_items, validate_capabilities
from job_io import (
    USAGE,
    parse_run_args,
    prepare_output_dir,
    resolve_output_dir,
    step_output_base,
    write_run_manifest,
    write_run_status,
    write_vtk_function,
    verify_run_artifacts,
)
from fc_profile import (
    FCProfileError,
    check_honored_outputs,
    normalize_units,
    parse_report_payload,
    required_output_fields,
    PARSE_NOT_SOLVED_BANNER,
)
from fc_assembly import (
    apply_cloads,
    check_solution_finite,
    collect_constrained_dofs,
    inp_element_to_cell,
    inp_node_to_geom,
    log_fc_stage,
    mem_profile_enabled,
    require_serial_fc,
    residual_on_constrained,
)
from fc_solver import SOLVER_LU, solve_linear
from job_io import SOLVER_AUTO, normalize_rtol, normalize_solver


def build_mesh_from_inp(model, debug=0):
    """Constroi a malha DOLFINx a partir do modelo INP (C3D4, C3D8, C3D10).

    C3D10 usa geometria quadratica (os nos de meio de aresta do .inp entram na
    geometria), entao arestas curvas sao respeitadas como no CalculiX.
    """
    if not model.elements:
        raise ValueError("Nenhum elemento encontrado no .inp")

    first_el = next(iter(model.elements.values()))
    etype = first_el.type.upper()

    node_ids = sorted(model.nodes.keys())
    node_map = {nid: i for i, nid in enumerate(node_ids)}

    coords = np.array([[model.nodes[nid].x, model.nodes[nid].y, model.nodes[nid].z]
                       for nid in node_ids], dtype=np.float64)

    if etype == "C3D10":
        # Ordem CalculiX/Abaqus do C3D10 = ordem VTK (VTK_QUADRATIC_TETRA).
        from dolfinx.io import VTKFile  # noqa: F401  (garante o modulo io)
        from dolfinx.cpp.io import perm_vtk
        from dolfinx.mesh import CellType
        perm = np.asarray(perm_vtk(CellType.tetrahedron, 10), dtype=np.int64)
    else:
        perm = None

    cells = []
    for el in model.elements.values():
        if el.type.upper() != etype:
            raise ValueError(f"Elementos mistos não suportados ({el.type} vs {etype})")
        try:
            nodes = [node_map[n] for n in el.nodes]
        except KeyError as exc:
            raise ValueError(f"Elemento {el.id} referencia no inexistente: {exc}") from exc
        if etype == "C3D8" and len(nodes) == 8:
            nodes = [nodes[0], nodes[1], nodes[3], nodes[2],
                     nodes[4], nodes[5], nodes[7], nodes[6]]
        elif perm is not None:
            nodes = [nodes[int(p)] for p in perm]
        cells.append(nodes)
    cells = np.array(cells, dtype=np.int64 if etype == "C3D10" else np.int32)

    degree = 1
    if etype == "C3D8":
        cell_name = "hexahedron"
    elif etype == "C3D4":
        cell_name = "tetrahedron"
    elif etype == "C3D10":
        cell_name = "tetrahedron"
        degree = 2
    else:
        raise ValueError(f"Tipo de elemento não suportado: {etype}")

    if etype == "C3D10":
        # So os nos usados pelos elementos entram na geometria quadratica.
        used = np.unique(cells.reshape(-1))
        remap = -np.ones(coords.shape[0], dtype=np.int64)
        remap[used] = np.arange(used.size)
        coords = coords[used]
        cells = remap[cells]

    if debug:
        print(f"Malha: {len(node_ids)} nós, {len(cells)} elementos ({etype})")

    coord_element = basix.ufl.element("Lagrange", cell_name, degree, shape=(coords.shape[1],))
    ufl_mesh = ufl.Mesh(coord_element)
    domain = mesh.create_mesh(MPI.COMM_WORLD, cells, ufl_mesh, coords)
    return domain, node_map


CCX_QUADRATURE_DEGREE = {"C3D8": 2, "C3D10": 2}


def element_dx(domain, etype: str):
    """Medida de integracao com a regra do CalculiX para o elemento.

    C3D8: Gauss 2x2x2 (grau 2 no hexaedro); C3D10: 4 pontos (grau 2 no tetraedro);
    C3D4: padrao (integrando constante, qualquer regra e exata). Em elemento distorcido
    ou curvo a regra muda o resultado, por isso segue a do CalculiX.
    """
    deg = CCX_QUADRATURE_DEGREE.get(etype.upper())
    if deg is None:
        return ufl.dx(domain=domain)
    return ufl.dx(domain=domain, metadata={"quadrature_degree": deg})


def element_degree(model) -> int:
    """Grau do espaco de deslocamento: 2 para C3D10, 1 para C3D4/C3D8."""
    first = next(iter(model.elements.values()))
    return 2 if first.type.upper() == "C3D10" else 1


def resolve_solver_choice(solver: str, ndofs: int, auto_lu_max_dofs):
    """Escolhe lu ou cg-gamg. ``auto``: abaixo do limite da configuracao -> lu, senao cg-gamg."""
    if solver != SOLVER_AUTO:
        return solver, None
    if auto_lu_max_dofs is None:
        raise ValueError("--solver auto exige --auto-lu-max-dofs (limite definido na configuracao)")
    chosen = "lu" if ndofs < int(auto_lu_max_dofs) else "cg-gamg"
    return chosen, {
        "requested": "auto",
        "n_dofs": int(ndofs),
        "auto_lu_max_dofs": int(auto_lu_max_dofs),
        "chosen": chosen,
        "rule": "n_dofs < auto_lu_max_dofs -> lu; senao cg-gamg",
    }


class StaticContext:
    """Malha, espacos, materiais e forma bilinear: montados uma vez para todos os passos."""

    def __init__(self, model, debug=0, stages=None, t0=None):
        t0 = t0 if t0 is not None else time.perf_counter()
        self.t0 = t0
        self.stages = stages
        domain, node_map = build_mesh_from_inp(model, debug=debug)
        log_fc_stage("mesh", t0, stages)
        domain.topology.create_connectivity(0, domain.topology.dim)
        domain.topology.create_connectivity(domain.topology.dim, 0)
        self.domain = domain
        self.nid_to_geom = inp_node_to_geom(domain, model)
        log_fc_stage("inp_node_to_geom", t0, stages)
        self.eid_to_cell = inp_element_to_cell(domain, model, self.nid_to_geom)
        log_fc_stage("inp_element_to_cell", t0, stages)
        if debug:
            print(f"Mapa INP->geom: {len(self.nid_to_geom)} nos; INP->celula: {len(self.eid_to_cell)} elems")
        self.degree = element_degree(model)
        self.etype = next(iter(model.elements.values())).type.upper()
        self.V = fem.functionspace(domain, ("Lagrange", self.degree, (domain.geometry.dim,)))
        log_fc_stage("functionspace", t0, stages)

        # Materiais por celula (DG-0), via *SOLID SECTION, sem material padrao.
        mats = materials_by_element_strict(model, "elastic")
        Q = fem.functionspace(domain, ("DG", 0))
        E_func = fem.Function(Q)
        nu_func = fem.Function(Q)
        ncell = E_func.x.array.size
        for eid, mat in mats.items():
            cell_idx = self.eid_to_cell[eid]
            if cell_idx >= ncell:
                raise RuntimeError("Indice de celula fora do espaco DG0 (serial esperado).")
            E_func.x.array[cell_idx] = mat.E
            nu_func.x.array[cell_idx] = mat.nu
        E_func.x.scatter_forward()
        nu_func.x.scatter_forward()
        log_fc_stage("materials", t0, stages)
        if debug:
            print(f"Materiais aplicados: E min={E_func.x.array.min():.0f}, max={E_func.x.array.max():.0f}")
        self.E = E_func
        self.nu = nu_func
        mu = E_func / (2.0 * (1.0 + nu_func))
        lmbda = E_func * nu_func / ((1.0 + nu_func) * (1.0 - 2.0 * nu_func))

        def epsilon(u):
            return ufl.sym(ufl.grad(u))

        def sigma(u):
            return lmbda * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2.0 * mu * epsilon(u)

        self.epsilon = epsilon
        self.sigma = sigma
        u = ufl.TrialFunction(self.V)
        self.v = ufl.TestFunction(self.V)
        self.dx = element_dx(domain, self.etype)
        a = ufl.inner(sigma(u), epsilon(self.v)) * self.dx
        L_form = ufl.dot(fem.Constant(domain, default_scalar_type((0.0, 0.0, 0.0))), self.v) * self.dx
        self.a_form = fem.form(a)
        self.L_compiled = fem.form(L_form)
        self.ndofs = int(self.V.dofmap.index_map.size_global * self.V.dofmap.index_map_bs)


def run_linear_static_step(model, step, output_base: str, debug: int = 0, profile: str = None,
                           solver: str = "lu", rtol: float = 1e-10, ctx: "StaticContext" = None,
                           auto_lu_max_dofs=None):
    """Um passo de estatica linear. ``step`` e o estado efetivo do passo (effective_steps)."""
    solver = normalize_solver(solver)
    rtol = normalize_rtol(rtol, solver)
    require_serial_fc(MPI.COMM_WORLD, True)
    stages = [] if mem_profile_enabled() else None
    if ctx is None:
        ctx = StaticContext(model, debug=debug, stages=stages)
    t0 = ctx.t0
    stages = ctx.stages
    domain, V = ctx.domain, ctx.V
    solver_used, auto_info = resolve_solver_choice(solver, ctx.ndofs, auto_lu_max_dofs)

    boundary = step.boundary if step.boundary is not None else model.boundary
    bcs, constrained_rows = collect_constrained_dofs(V, model, True, boundary=boundary)
    log_fc_stage("collect_constrained_dofs", t0, stages)
    if not constrained_rows:
        raise RuntimeError(
            "Recusado: sistema sem DOFs prescritos (*BOUNDARY). "
            "Ha modos de corpo rigido; uma fatoracao LU que nao acusa "
            "singularidade nao conta como sucesso."
        )

    A = assemble_matrix(ctx.a_form, bcs=bcs)
    A.assemble()
    log_fc_stage("assemble_A", t0, stages)
    b = assemble_vector(ctx.L_compiled)
    log_fc_stage("assemble_b", t0, stages)

    applied = apply_cloads(V, b, model, step, True)
    log_fc_stage("apply_cloads", t0, stages)

    if step.dloads:
        from fc_pressure import apply_pressure_loads
        p_applied = apply_pressure_loads(ctx, b, model, step)
        for key in ("Fx", "Fy", "Fz"):
            applied[key] += p_applied["F"][key]
        applied["pressure"] = p_applied
        log_fc_stage("apply_dloads", t0, stages)

    f_ext = b.copy()

    apply_lifting(b, [ctx.a_form], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)

    uh = fem.Function(V)
    solver_info = solve_linear(
        A, b, uh, V, solver=solver_used, rtol=rtol, debug=debug,
    )
    if auto_info:
        solver_info["auto"] = auto_info
    log_fc_stage("solve", t0, stages)
    uh.name = "displacement"

    write_vtk_function(domain.comm, uh, output_base, debug=debug)
    fields = _write_fc_linear_fields(
        domain, uh, ctx.v, ctx.sigma, ctx.epsilon, f_ext, ctx.a_form, model, output_base, debug,
        constrained_rows=constrained_rows, applied=applied, loads=step.loads, ctx=ctx,
    )
    fields["solver_info"] = solver_info
    extra = fields
    log_fc_stage("fc_fields_rf", t0, stages)
    _check_fc_force_balance(extra)
    try:
        A.destroy()
        b.destroy()
        f_ext.destroy()
    except Exception:
        pass

    norma = np.linalg.norm(uh.x.array)
    extra["u_max"] = float(np.max(np.linalg.norm(uh.x.array.reshape(-1, 3), axis=1))) if uh.x.array.size else 0.0
    extra["analysis"] = "static"
    if debug:
        print(f"Norma: {norma:.6e}")
    return uh, extra


def run_from_inp(inp_path: str, output_dir: str = "/app/output", debug: int = 0,
                 profile: str = None, units: str = None, parse_only: bool = False,
                 run_id: str = None, solver: str = "lu", rtol: float = 1e-10,
                 restrictions=None, auto_lu_max_dofs=None):
    """Executa cada passo e grava a saida somente em output_dir."""
    if not os.path.exists(inp_path):
        print(f"ERRO: arquivo nao encontrado: {inp_path}")
        sys.exit(1)

    profile_fc = (profile or "").strip().lower() == "fc"
    units_norm = None
    if profile_fc and not units:
        raise FCProfileError("Perfil FC exige --units (use mm-N-MPa-t-s).")
    if units:
        units_norm = normalize_units(units)
    restrictions = restrictions or {}

    output_dir = prepare_output_dir(resolve_output_dir(output_dir))
    run_id = run_id or str(uuid.uuid4())
    write_run_status(output_dir, {
        "status": "started",
        "run_id": run_id,
        "solver_run": not parse_only,
        "solver": normalize_solver(solver),
        "rtol": normalize_rtol(rtol, normalize_solver(solver)),
        "note": "run em andamento; manifesto antigo neste diretorio nao e sucesso desta run",
    })

    model = parse_inp(inp_path, debug=debug, profile=profile)
    validate_capabilities(model)
    apply_restrictions(model, restrictions)
    if debug:
        print(summary(model))
        print("---")

    if parse_only:
        payload = parse_report_payload(model, os.path.abspath(inp_path))
        payload["profile"] = profile or "generico"
        payload["run_id"] = run_id
        payload["units"] = units_norm
        payload["restrictions"] = restrictions
        report_path = os.path.join(output_dir, "parse_report.json")
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(PARSE_NOT_SOLVED_BANNER)
        print("Parse report:", report_path)
        write_run_status(output_dir, {
            "status": "parse_ok_not_solved",
            "run_id": run_id,
            "solver_run": False,
            "parse_report": report_path,
        })
        return {"parse_only": True, "model": model, "report": payload}

    steps = effective_steps(model)
    results = []
    ctx = None
    for i, step in enumerate(steps):
        step_num = i + 1
        output_base = step_output_base(output_dir, inp_path, step_num)
        if debug:
            print(f"\n=== Step {step_num} ===")
        if step.analysis_type == "heat":
            from run_thermal import run_heat_steady
            result, extra = run_heat_steady(
                model, step, output_base, debug=debug, solver=solver, rtol=rtol,
                auto_lu_max_dofs=auto_lu_max_dofs,
            )
        else:
            if ctx is None:
                ctx = StaticContext(model, debug=debug, stages=[] if mem_profile_enabled() else None)
            result, extra = run_linear_static_step(
                model, step, output_base, debug, profile=profile,
                solver=solver, rtol=rtol, ctx=ctx, auto_lu_max_dofs=auto_lu_max_dofs,
            )
        extra["step"] = step_num
        extra["output_base"] = os.path.basename(output_base)
        results.append((result, extra))

    _finalize_fc_run(model, output_dir, inp_path, units_norm, results, debug, run_id=run_id,
                     profile=profile, restrictions=restrictions)
    return results

def _cell_average_tensor(ctx, expr, name: str):
    """Media aritmetica do tensor nos pontos de integracao do CalculiX (DG0).

    C3D8: Gauss 2x2x2; C3D10: 4 pontos (pesos iguais nas duas regras). Dividir o
    integrando por |det J| deixa so os pesos de referencia, entao o resultado e a
    media simples dos valores nos pontos (o que o *EL PRINT do ccx lista), tambem
    em elemento distorcido ou curvo.
    """
    from dolfinx import fem
    W = fem.functionspace(ctx.domain, ("DG", 0, (3, 3)))
    w = ufl.TestFunction(W)
    one = fem.functionspace(ctx.domain, ("DG", 0))
    q = ufl.TestFunction(one)
    dx = element_dx(ctx.domain, ctx.etype)
    inv_j = 1.0 / abs(ufl.JacobianDeterminant(ctx.domain))
    num = assemble_vector(fem.form(inv_j * ufl.inner(expr, w) * dx))
    vol = assemble_vector(fem.form(inv_j * q * dx))
    func = fem.Function(W)
    func.name = name
    vals = np.asarray(num.array).reshape(-1, 9) / np.asarray(vol.array).reshape(-1, 1)
    func.x.array[:] = vals.reshape(-1)
    func.x.scatter_forward()
    return func


def _write_fc_linear_fields(domain, uh, v, sigma, epsilon, f_ext, a_form, model, output_base, debug,
                            constrained_rows=None, applied=None, loads=None, ctx=None):
    """Grava U, S, principais, von Mises, E e RF do perfil FC. Falha se VTK falhar."""
    import json
    import os
    from dolfinx import fem
    from fc_fields import (
        interpolate_cell_scalar,
        interpolate_cell_tensor,
        interpolate_cell_vector,
        principals_algebraic,
        tensor_array_from_function,
        write_functions_vtk,
    )

    uh.name = "displacement"
    strain_ufl = epsilon(uh)
    cauchy_ufl = sigma(uh)
    if ctx is None or ctx.etype == "C3D4":
        # C3D4 P1: tensao constante na celula; interpolacao DG0 exata.
        cauchy_f = interpolate_cell_tensor(domain, cauchy_ufl, "cauchy_stress")
        strain_f = interpolate_cell_tensor(domain, strain_ufl, "strain")
        cell_location = "cell-DG0-constant-C3D4-P1"
    else:
        cauchy_f = _cell_average_tensor(ctx, cauchy_ufl, "cauchy_stress")
        strain_f = _cell_average_tensor(ctx, strain_ufl, "strain")
        cell_location = "cell-DG0-average-of-integration-points-{0}".format(ctx.etype)
    sigma_cells = tensor_array_from_function(cauchy_f)
    # Simetria residual ja e aplicada em principals_algebraic.
    principals, vm = principals_algebraic(sigma_cells)
    prin_f = interpolate_cell_vector(domain, principals, "principal_stress")
    vm_f = interpolate_cell_scalar(domain, vm, "von_mises")

    cell_pvd = write_functions_vtk(
        domain.comm,
        [cauchy_f, strain_f, prin_f, vm_f],
        output_base + "_cell",
    )

    # R = K u - f_ext (assemble_vector esparso), somente nos DOFs prescritos (livres ~0).
    dx_el = ctx.dx if ctx is not None else ufl.dx
    f_int = assemble_vector(fem.form(ufl.inner(sigma(uh), epsilon(v)) * dx_el))
    try:
        f_int.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    except Exception:
        pass

    constrained_rows = constrained_rows or []
    applied = applied or {"Fx": 0.0, "Fy": 0.0, "Fz": 0.0}
    coords_by_nid = {nid: (n.x, n.y, n.z) for nid, n in model.nodes.items()}
    rx = residual_on_constrained(f_int, f_ext, constrained_rows, coords_by_nid)

    V = uh.function_space
    rf = fem.Function(V)
    rf.name = "reaction_force"
    rf.x.array[:] = 0.0
    mlen = min(rf.x.array.size, rx["rf_constrained"].size)
    rf.x.array[:mlen] = rx["rf_constrained"][:mlen]
    rf.x.scatter_forward()
    rf_pvd = write_functions_vtk(domain.comm, [rf], output_base + "_RF")

    totals = rx["totals"]
    sum_rf = {
        "Fx": totals["RFx"] + applied["Fx"],
        "Fy": totals["RFy"] + applied["Fy"],
        "Fz": totals["RFz"] + applied["Fz"],
    }
    # Momento das cargas aplicadas em torno da origem.
    mom_f = {"Mx": 0.0, "My": 0.0, "Mz": 0.0}
    if loads is None:
        loads = model.steps[0].loads if model.steps else []
    for cl in loads:
        if cl.node not in model.nodes:
            continue
        n = model.nodes[cl.node]
        fx = cl.value if cl.dof == 1 else 0.0
        fy = cl.value if cl.dof == 2 else 0.0
        fz = cl.value if cl.dof == 3 else 0.0
        mom_f["Mx"] += n.y * fz - n.z * fy
        mom_f["My"] += n.z * fx - n.x * fz
        mom_f["Mz"] += n.x * fy - n.y * fx
    mom_sum = {
        "Mx": rx["moment_R_origin"]["Mx"] + mom_f["Mx"],
        "My": rx["moment_R_origin"]["My"] + mom_f["My"],
        "Mz": rx["moment_R_origin"]["Mz"] + mom_f["Mz"],
    }
    force_char = 0.0
    moment_char = 0.0
    for cl in loads:
        if cl.node not in model.nodes:
            continue
        n = model.nodes[cl.node]
        fx = cl.value if cl.dof == 1 else 0.0
        fy = cl.value if cl.dof == 2 else 0.0
        fz = cl.value if cl.dof == 3 else 0.0
        force_char += abs(fx) + abs(fy) + abs(fz)
        moment_char += abs(n.y * fz - n.z * fy) + abs(n.z * fx - n.x * fz) + abs(n.x * fy - n.y * fx)
    for ent in rx.get("entries") or []:
        val = float(ent.get("R", 0.0))
        comp = int(ent.get("component", 0))
        nid = int(ent.get("inp_node", 0))
        if nid not in model.nodes:
            continue
        n = model.nodes[nid]
        fx = val if comp == 0 else 0.0
        fy = val if comp == 1 else 0.0
        fz = val if comp == 2 else 0.0
        force_char += abs(val)
        moment_char += abs(n.y * fz - n.z * fy) + abs(n.z * fx - n.x * fz) + abs(n.x * fy - n.y * fx)

    react_path = output_base + "_reactions.json"
    payload = {
        "totals": totals,
        "applied_totals": applied,
        "sum_R_plus_F": sum_rf,
        "moment_R_origin": rx["moment_R_origin"],
        "moment_F_origin": mom_f,
        "moment_R_plus_F_origin": mom_sum,
        "n_constrained_dofs": rx["n_constrained_dofs"],
        "residual_free_dofs_max_abs": rx["residual_free_dofs_max_abs"],
        "force_char": force_char,
        "moment_char": moment_char,
        "entries": rx["entries"],
        "note": (
            "RF = (K_full u - f_ext) apenas em DOFs prescritos. "
            "Equilibrio de forca/momento: sum(R+F) e momentos em torno da origem ~0. "
            "Nao e aceite de roda/Luiz."
        ),
    }
    with open(react_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")

    if debug:
        print("FC campos celula:", cell_pvd)
        print("FC RF:", totals, "R+F:", sum_rf, "free residual max:", rx["residual_free_dofs_max_abs"])

    pressure = (applied or {}).get("pressure")
    if pressure:
        # Momento da pressao (somado como forcas nodais consistentes).
        for key in ("Mx", "My", "Mz"):
            mom_f[key] += pressure["M"][key]
        mom_sum = {k: rx["moment_R_origin"][k] + mom_f[k] for k in ("Mx", "My", "Mz")}
        force_char += pressure["force_char"]
        moment_char += pressure["moment_char"]
        payload["moment_F_origin"] = mom_f
        payload["moment_R_plus_F_origin"] = mom_sum
        payload["force_char"] = force_char
        payload["moment_char"] = moment_char
        payload["pressure"] = {k: pressure[k] for k in ("F", "n_faces")}
        with open(react_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")

    degree = getattr(ctx, "degree", 1)
    honored = ["U", "S", "E", "PRINCIPAL", "VON_MISES", "RF"]
    artifacts = [
        {"path": output_base + ".pvd", "field": "U", "location": "nodal-CG{0}".format(degree),
         "space": "Lagrange-P{0}".format(degree)},
        {"path": cell_pvd, "field": "S,E,PRINCIPAL,VON_MISES", "location": cell_location,
         "tensor_order": "row-major XX,XY,XZ,YX,YY,YZ,ZX,ZY,ZZ",
         "principals": "algebraic sigma1>=sigma2>=sigma3 with sign; tensor symmetrized before eig",
         "note": "nodal smoothed does not replace cell field"},
        {"path": rf_pvd, "field": "RF", "location": "nodal-constrained-dofs-only"},
        {"path": react_path, "field": "RF_totals", "location": "json"},
    ]
    return {
        "honored": honored,
        "artifacts": artifacts,
        "reactions": payload,
        "applied": applied,
        "stress_cells": sigma_cells,
        "principals": principals,
        "von_mises": vm,
    }


def _check_fc_force_balance(extra, rel=1e-5, abs_tol=1e-6):
    """Recusa sucesso se R+F (e momentos) nao fecharem no residual do proprio solver.

    Tolerancia numerica de consistencia (LU / residual), nao aceite Luiz/GateV.
    Forca/momento resultantes nulos (binario puro) usam a escala caracteristica
    das parcelas (|F_i| e |r x F_i| + |r x R_i|), nao o total que se cancela.
    """
    rx = extra.get("reactions") or {}
    sum_rf = rx.get("sum_R_plus_F") or {}
    mom = rx.get("moment_R_plus_F_origin") or {}
    applied = rx.get("applied_totals") or extra.get("applied") or {}
    totals = rx.get("totals") or {}
    scale_f = max(
        [
            abs_tol,
            abs(float(applied.get("Fx", 0.0))),
            abs(float(applied.get("Fy", 0.0))),
            abs(float(applied.get("Fz", 0.0))),
            abs(float(totals.get("RFx", 0.0))),
            abs(float(totals.get("RFy", 0.0))),
            abs(float(totals.get("RFz", 0.0))),
            abs(float(rx.get("force_char", 0.0))),
        ]
    )
    for key in ("Fx", "Fy", "Fz"):
        val = abs(float(sum_rf.get(key, 0.0)))
        if val > max(abs_tol, rel * scale_f):
            raise RuntimeError(
                "Equilibrio de forca nao fechou no perfil FC ({0}={1}, escala={2}). "
                "Nao e sucesso.".format(key, sum_rf.get(key), scale_f)
            )
    scale_m = max(
        [
            abs_tol,
            scale_f,  # fallback se nao houver binario
            abs(float(rx.get("moment_char", 0.0))),
            abs(float((rx.get("moment_F_origin") or {}).get("Mx", 0.0))),
            abs(float((rx.get("moment_F_origin") or {}).get("My", 0.0))),
            abs(float((rx.get("moment_F_origin") or {}).get("Mz", 0.0))),
            abs(float((rx.get("moment_R_origin") or {}).get("Mx", 0.0))),
            abs(float((rx.get("moment_R_origin") or {}).get("My", 0.0))),
            abs(float((rx.get("moment_R_origin") or {}).get("Mz", 0.0))),
        ]
    )
    for key in ("Mx", "My", "Mz"):
        val = abs(float(mom.get(key, 0.0)))
        if val > max(abs_tol, rel * scale_m):
            raise RuntimeError(
                "Equilibrio de momento nao fechou no perfil FC ({0}={1}, escala={2}). "
                "Nao e sucesso.".format(key, mom.get(key), scale_m)
            )


def _stack_versions():
    info = {"python": sys.version.split()[0]}
    for name in ("dolfinx", "ufl", "basix", "ffcx", "petsc4py", "mpi4py", "numpy"):
        try:
            mod = __import__(name)
            info[name] = getattr(mod, "__version__", "?")
        except Exception:
            info[name] = "ausente"
    return info


def _finalize_fc_run(model, output_dir, inp_path, units_norm, results, debug, run_id=None,
                     profile=None, restrictions=None):
    rec = model.fc_record
    honored = None
    artifacts = []
    reactions = None
    solver_info = None
    steps_info = []
    for _res, extra in results:
        h = list(extra.get("honored") or [])
        honored = h if honored is None else [f for f in honored if f in h]
        artifacts.extend(extra.get("artifacts") or [])
        reactions = extra.get("reactions") or reactions
        solver_info = extra.get("solver_info") or solver_info
        rx = extra.get("reactions") or {}
        steps_info.append({
            "step": extra.get("step"),
            "analysis": extra.get("analysis"),
            "output_base": extra.get("output_base"),
            "honored": h,
            "u_max": extra.get("u_max"),
            "t_min": extra.get("t_min"),
            "t_max": extra.get("t_max"),
            "reaction_totals": rx.get("totals"),
            "applied_totals": rx.get("applied_totals"),
            "solver": extra.get("solver_info"),
        })
    honored = honored or []

    analysis = {s.get("analysis") for s in steps_info}
    if analysis == {"heat"}:
        requested = ["NT"] + [f for r in (rec.requests if rec else []) for f in r.fields if f not in ("NT",)]
    else:
        requested = required_output_fields(rec) if rec else ["U", "S", "PRINCIPAL", "VON_MISES"]
    check_honored_outputs(requested, honored)

    payload = {
        "status": "ok",
        "run_id": run_id,
        "solver_run": True,
        "profile": profile or "generico",
        "restrictions": restrictions or {},
        "units": units_norm if units_norm else "nao declaradas (--units ausente); valores nas unidades do deck",
        "steps": steps_info,
        "inp": os.path.abspath(inp_path),
        "start_time": 0.0,
        "n_steps": len(model.steps),
        "element_types": sorted({el.type.upper() for el in model.elements.values()}),
        "n_nodes": len(model.nodes),
        "n_elements": len(model.elements),
        "materials": {
            name: {"E": mat.E, "nu": mat.nu} for name, mat in model.materials.items()
        },
        "ignored_cards": [
            {"keyword": it.keyword, "reason": it.reason, "line": it.line, "detail": it.detail}
            for it in (rec.ignored if rec else [])
        ] + [it for it in ignored_items(model)
             if not any(x.line == it["line"] for x in (rec.ignored if rec else []))],
        "output_requests": [
            {"kind": r.kind, "fields": r.fields, "nset": r.nset, "totals_only": r.totals_only, "line": r.line}
            for r in (rec.requests if rec else [])
        ],
        "honored_fields": honored,
        "field_contract": {
            "U": "nodal Lagrange P1",
            "S": "Cauchy cell DG0, constant on C3D4 P1; not nodal-smoothed; symmetrized before principals",
            "E": "infinitesimal strain cell DG0",
            "PRINCIPAL": "algebraic sigma1>=sigma2>=sigma3 with sign, cell DG0",
            "VON_MISES": "cell DG0",
            "RF": "residual K_full u - f_ext at constrained DOFs only, plus force/moment totals",
        },
        "static_increments": rec.static_increments if rec else None,
        "reactions": reactions,
        "solver": solver_info,
        "solver_versions": _stack_versions(),
        "image": {
            "name_requested": os.environ.get("FENICS_IMAGE", "edss/fenicsx:dev"),
            "note": (
                "Versoes acima sao as importadas nesta execucao. "
                "Image ID local (docker image inspect .Id) identifica o artefato nesta maquina "
                "e NAO e um RepoDigest verificavel de registry, nem referencia de pull. "
                "Nao usar <repo>:<tag>@sha256:<imageId> como pin de registry."
            ),
        },
        "artifacts": artifacts,
        "limits": {
            "wheel_validation": False,
            "independent_benchmark": "axial+compressao analitica em fixture sintetica; GateV pendente",
            "material_plan_70000": "nao corrigido automaticamente se o .inp trouxer outro E",
            "mpi": "serial only (MPI size=1)",
        },
    }
    manifest_path = write_run_manifest(output_dir, payload)
    required = [os.path.basename(a["path"]) for a in artifacts if a.get("path")]
    verify_run_artifacts(manifest_path, required)
    status_body = {
        "status": "ok",
        "run_id": run_id,
        "solver_run": True,
        "manifest": manifest_path,
    }
    if solver_info:
        status_body["solver"] = solver_info
    write_run_status(output_dir, status_body)
    if debug:
        print("Manifesto:", manifest_path)


if __name__ == "__main__":
    try:
        args = parse_run_args(sys.argv[1:])
    except ValueError as exc:
        print(f"ERRO: {exc}")
        print(USAGE)
        sys.exit(2)
    if not args.inp_file:
        print(USAGE)
        sys.exit(1)
    output_dir = resolve_output_dir(args.output_dir)
    run_id = str(uuid.uuid4())
    profile_fc = (args.profile or "").strip().lower() == "fc"
    try:
        run_from_inp(
            args.inp_file,
            output_dir=output_dir,
            debug=args.debug,
            profile=args.profile,
            units=args.units,
            parse_only=args.parse_only,
            run_id=run_id,
            solver=args.solver,
            rtol=args.rtol,
            restrictions=args.restrictions,
            auto_lu_max_dofs=args.auto_lu_max_dofs,
        )
    except FCProfileError as exc:
        print("ERRO INP: {0}".format(exc))
        if True:
            try:
                write_run_status(prepare_output_dir(output_dir), {
                    "status": "failed",
                    "run_id": run_id,
                    "solver_run": False if args.parse_only else True,
                    "error": str(exc),
                    "error_class": "FCProfileError",
                    "note": "falha: manifesto de sucesso anterior neste diretorio nao vale para esta run",
                })
            except Exception:
                pass
        sys.exit(2)
    except RuntimeError as exc:
        print("ERRO: {0}".format(exc))
        if True:
            try:
                failed = {
                    "status": "failed",
                    "run_id": run_id,
                    "solver_run": True,
                    "error": str(exc),
                    "error_class": "RuntimeError",
                    "note": "falha: manifesto de sucesso anterior neste diretorio nao vale para esta run",
                }
                if getattr(exc, "solver_info", None):
                    failed["solver"] = exc.solver_info
                else:
                    failed["solver"] = {"type": getattr(args, "solver", None), "rtol": getattr(args, "rtol", None)}
                write_run_status(prepare_output_dir(output_dir), failed)
            except Exception:
                pass
        sys.exit(3)
    except Exception as exc:
        print("ERRO: {0}: {1}".format(type(exc).__name__, exc))
        if True:
            try:
                write_run_status(prepare_output_dir(output_dir), {
                    "status": "failed",
                    "run_id": run_id,
                    "solver_run": True,
                    "error": "{0}: {1}".format(type(exc).__name__, exc),
                    "error_class": type(exc).__name__,
                    "note": "falha: manifesto de sucesso anterior neste diretorio nao vale para esta run",
                })
            except Exception:
                pass
        sys.exit(3)
