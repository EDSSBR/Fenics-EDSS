"""*DLOAD de pressao uniforme (P1..P6) nas faces dos elementos, numeracao do CalculiX.

Faces (indices dos nos do elemento, base 0), CalculiX 2.23:
  C3D8:        P1 1-2-3-4, P2 5-8-7-6, P3 1-5-6-2, P4 2-6-7-3, P5 3-7-8-4, P6 4-8-5-1
  C3D4/C3D10:  P1 1-2-3,   P2 1-4-2,   P3 2-4-3,   P4 3-4-1  (C3D10: os nos de meio
               de aresta da face entram pela geometria quadratica)
Pressao positiva empurra a face para dentro do elemento: t = -p n (n normal externa).
Vetor de carga consistente: f_i = integral de N_i t na face, com a geometria do
elemento (curva no C3D10). Face interna (compartilhada por dois elementos) e outros
tipos de *DLOAD (GRAV, CENTRIF, BX...) nao sao implementados: erro explicito.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from fc_profile import FCProfileError
from inp_capabilities import InpNotImplementedError

FACES = {
    "C3D8": {"P1": (0, 1, 2, 3), "P2": (4, 7, 6, 5), "P3": (0, 4, 5, 1),
             "P4": (1, 5, 6, 2), "P5": (2, 6, 7, 3), "P6": (3, 7, 4, 0)},
    "C3D4": {"P1": (0, 1, 2), "P2": (0, 3, 1), "P3": (1, 3, 2), "P4": (2, 3, 0)},
}
FACES["C3D10"] = FACES["C3D4"]
# Regra de integracao na face (grau da quadratura): a mesma familia usada no volume.
FACE_QUADRATURE_DEGREE = {"C3D8": 2, "C3D10": 2}


def _facet_index(ctx):
    """Para cada celula: chave (geometria dos vertices, ordenada) -> faceta."""
    from dolfinx import mesh as dmesh
    domain = ctx.domain
    top = domain.topology
    tdim = top.dim
    fdim = tdim - 1
    top.create_connectivity(tdim, fdim)
    top.create_connectivity(fdim, tdim)
    top.create_connectivity(fdim, 0)
    nverts = top.index_map(0).size_local + top.index_map(0).num_ghosts
    v2g = np.asarray(dmesh.entities_to_geometry(domain, 0, np.arange(nverts, dtype=np.int32), False)).reshape(-1)
    return top.connectivity(tdim, fdim), top.connectivity(fdim, tdim), top.connectivity(fdim, 0), v2g


def pressure_facets(ctx, model, step) -> Tuple[np.ndarray, np.ndarray]:
    """(facetas, pressao) do estado efetivo do passo; erro explicito no que nao cabe."""
    etype = ctx.etype
    faces = FACES.get(etype)
    if faces is None:
        raise InpNotImplementedError("*DLOAD em elemento {0} nao implementado.".format(etype))
    c2f, f2c, f2v, v2g = _facet_index(ctx)
    acc: Dict[int, float] = {}
    for dl in step.dloads:
        if dl.face not in faces:
            raise InpNotImplementedError(
                "*DLOAD tipo {0} nao implementado para {1} (so pressao {2}).".format(
                    dl.face, etype, sorted(faces)))
        eid = int(dl.elset)
        el = model.elements[eid]
        try:
            want = tuple(sorted(ctx.nid_to_geom[el.nodes[i]] for i in faces[dl.face]))
        except KeyError as exc:
            raise FCProfileError("*DLOAD no elemento {0}: no sem geometria {1}".format(eid, exc)) from exc
        cell = ctx.eid_to_cell[eid]
        hit = None
        for f in c2f.links(cell):
            if tuple(sorted(int(v2g[v]) for v in f2v.links(f))) == want:
                hit = int(f)
                break
        if hit is None:
            raise FCProfileError("*DLOAD elemento {0} face {1}: face nao encontrada na malha.".format(eid, dl.face))
        if len(f2c.links(hit)) != 1:
            raise InpNotImplementedError(
                "*DLOAD elemento {0} face {1}: face interna (compartilhada por dois elementos) "
                "nao implementada.".format(eid, dl.face))
        acc[hit] = acc.get(hit, 0.0) + float(dl.value)
    facets = np.array(sorted(acc), dtype=np.int32)
    return facets, np.array([acc[int(f)] for f in facets], dtype=np.float64)


def apply_pressure_loads(ctx, b, model, step):
    import ufl
    from dolfinx import fem
    from dolfinx import mesh as dmesh
    from dolfinx.fem.petsc import assemble_vector
    from petsc4py import PETSc

    facets, pvals = pressure_facets(ctx, model, step)
    zero = {"F": {"Fx": 0.0, "Fy": 0.0, "Fz": 0.0}, "M": {"Mx": 0.0, "My": 0.0, "Mz": 0.0},
            "force_char": 0.0, "moment_char": 0.0, "n_faces": 0}
    if facets.size == 0:
        return zero
    domain = ctx.domain
    fdim = domain.topology.dim - 1
    # Um rotulo por valor de pressao distinto.
    uniq, tag_of = np.unique(pvals, return_inverse=True)
    tags = (tag_of + 1).astype(np.int32)
    mt = dmesh.meshtags(domain, fdim, facets, tags)
    deg = FACE_QUADRATURE_DEGREE.get(ctx.etype)
    md = {"quadrature_degree": deg} if deg is not None else {}
    ds = ufl.Measure("ds", domain=domain, subdomain_data=mt, metadata=md)
    n = ufl.FacetNormal(domain)
    v = ctx.v
    form = None
    for k, p in enumerate(uniq):
        term = -float(p) * ufl.dot(n, v) * ds(k + 1)
        form = term if form is None else form + term
    bp = assemble_vector(fem.form(form))
    bp.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    V = ctx.V
    nloc = V.dofmap.index_map.size_local
    f = np.asarray(bp.array)[: nloc * 3].reshape(-1, 3)
    X = np.asarray(V.tabulate_dof_coordinates())[:nloc]
    F = f.sum(axis=0)
    M = np.cross(X, f).sum(axis=0)
    b.axpy(1.0, bp)
    return {
        "F": {"Fx": float(F[0]), "Fy": float(F[1]), "Fz": float(F[2])},
        "M": {"Mx": float(M[0]), "My": float(M[1]), "Mz": float(M[2])},
        "force_char": float(np.abs(f).sum()),
        "moment_char": float((np.linalg.norm(X, axis=1)[:, None] * np.abs(f)).sum()),
        "n_faces": int(facets.size),
    }
