"""C3D10 (tetraedro quadratico) e regra de integracao do CalculiX (passo 5, 21/11/2026).

Os decks sao gerados aqui (grade hexaedrica -> 6 tetraedros por divisao de Kuhn,
nos de meio de aresta na ordem CalculiX: 5:(1,2) 6:(2,3) 7:(3,1) 8:(1,4) 9:(2,4)
10:(3,4)). As referencias sao do CalculiX 2.23 (condaforge-2.23) rodado no Nitro em
21/11/2026 com estes mesmos decks; a comparacao usa a precisao de impressao do
.frd/.dat, nao um criterio de aceite.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

KUHN = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]
EDGES = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))


def _vol6(a, b, c, d):
    u = [b[i] - a[i] for i in range(3)]
    v = [c[i] - a[i] for i in range(3)]
    w = [d[i] - a[i] for i in range(3)]
    return u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0]) + u[2] * (v[0] * w[1] - v[1] * w[0])


def _grid(nx, ny, nz, L, B, H):
    def nid(i, j, k):
        return 1 + i + (nx + 1) * (j + (ny + 1) * k)
    nodes = {nid(i, j, k): (L * i / nx, B * j / ny, H * k / nz)
             for k in range(nz + 1) for j in range(ny + 1) for i in range(nx + 1)}
    hexes = [(i, [nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k), nid(i, j + 1, k),
                  nid(i, j, k + 1), nid(i + 1, j, k + 1), nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1)])
             for k in range(nz) for j in range(ny) for i in range(nx)]
    return nodes, hexes, nid


def _tets(nodes, hexes):
    out = []
    for ix, h in hexes:
        for t in KUHN:
            v = [h[q] for q in t]
            if _vol6(*[nodes[n] for n in v]) < 0:
                v[1], v[2] = v[2], v[1]
            out.append((ix, v))
    return out


def _quad(nodes, tets):
    nodes = dict(nodes)
    nxt = max(nodes) + 1
    mid = {}
    out = []
    for ix, v in tets:
        extra = []
        for a, b in EDGES:
            key = tuple(sorted((v[a], v[b])))
            if key not in mid:
                pa, pb = nodes[key[0]], nodes[key[1]]
                mid[key] = nxt
                nodes[nxt] = tuple((pa[i] + pb[i]) / 2 for i in range(3))
                nxt += 1
            extra.append(mid[key])
        out.append((ix, v + extra))
    return nodes, out


def _deck(path, etype, nodes, els, mats, fixed, steps, extra_nodes=""):
    with open(path, "w") as f:
        f.write("*NODE\n")
        for n, (x, y, z) in sorted(nodes.items()):
            f.write(f"{n}, {x:.10g}, {y:.10g}, {z:.10g}\n")
        f.write(extra_nodes)
        f.write(f"*ELEMENT, TYPE={etype}, ELSET=Eall\n")
        for eid, (_, c) in enumerate(els, 1):
            s = [str(eid)] + [str(n) for n in c]
            f.write(", ".join(s[:16]) + ("\n" if len(s) <= 16 else ",\n" + ", ".join(s[16:]) + "\n"))
        for name, (E, nu, sel) in mats.items():
            ids = [str(e) for e, (ix, _) in enumerate(els, 1) if sel(ix)]
            f.write(f"*ELSET, ELSET=E_{name}\n")
            for i in range(0, len(ids), 16):
                f.write(", ".join(ids[i:i + 16]) + "\n")
            f.write(f"*MATERIAL, NAME={name}\n*ELASTIC\n{E}, {nu}\n*SOLID SECTION, ELSET=E_{name}, MATERIAL={name}\n")
        f.write("*NSET, NSET=MIKEFIX\n")
        for i in range(0, len(fixed), 16):
            f.write(", ".join(map(str, fixed[i:i + 16])) + "\n")
        f.write("*BOUNDARY\nMIKEFIX, 1, 3, 0.0\n")
        for st in steps:
            f.write("*STEP\n*STATIC\n*CLOAD\n")
            for n, d, v in st:
                f.write(f"{n}, {d}, {v}\n")
            f.write("*NODE FILE\nU\n*END STEP\n")


SB_NODES = {1: (0.0, 0.0, 0.0), 2: (0.333, 0.0, 0.0), 3: (0.667, 0.0, 0.0), 4: (1.0, 0.0, 0.0),
            5: (0.0, 0.2, 0.0), 6: (0.333, 0.2, 0.0), 7: (0.667, 0.2, 0.0), 8: (1.0, 0.2, 0.0),
            9: (0.0, 0.0, 0.2), 10: (0.333, 0.0, 0.2), 11: (0.667, 0.0, 0.2), 12: (1.0, 0.0, 0.2),
            13: (0.0, 0.2, 0.2), 14: (0.333, 0.2, 0.2), 15: (0.667, 0.2, 0.2), 16: (1.0, 0.2, 0.2)}
SB_HEX = [(0, [1, 2, 6, 5, 9, 10, 14, 13]), (1, [2, 3, 7, 6, 10, 11, 15, 14]), (2, [3, 4, 8, 7, 11, 12, 16, 15])]
SB_LOADS = [[(4, 3, -30.0), (8, 3, -30.0), (12, 3, -30.0), (16, 3, -30.0)]]
ACO = {"Aco": (210000, 0.3, lambda ix: True)}


def simple_beam_c3d10(path, extra_nodes=""):
    """Exemplo 0008 (simple_beam) com cada C3D8 trocado por 6 C3D10; mesmos BC e CLOAD."""
    qn, qe = _quad(SB_NODES, _tets(SB_NODES, SB_HEX))
    _deck(path, "C3D10", qn, qe, ACO, [1, 5, 9, 13], SB_LOADS, extra_nodes)


def beam_c3d10_curvo(path):
    """Viga 100x10x10 mm, 240 C3D10, aco/aluminio, 2 passos; nos de meio de aresta
    internos deslocados ate 0.3 mm em z (elementos curvos)."""
    nodes, hexes, nid = _grid(10, 2, 2, 100.0, 10.0, 10.0)
    qn, qe = _quad(nodes, _tets(nodes, hexes))
    used_mid = {n for _, c in qe for n in c[4:]}
    for n in used_mid:
        x, y, z = qn[n]
        if 0 < x < 100 and 0 < y < 10 and 0 < z < 10:
            qn[n] = (x, y, float("%.10g" % (z + 0.3 * ((n % 7) - 3) / 3.0)))
    fixed = sorted(n for n, (x, _, _) in qn.items() if abs(x) < 1e-12)
    tip = sorted(n for n, (x, _, _) in qn.items() if abs(x - 100.0) < 1e-12)
    midedge = [n for n in tip if n not in nodes and abs(qn[n][1] - 7.5) < 1e-12 and abs(qn[n][2] - 10.0) < 1e-12][0]
    mats = {"Aco": (210000, 0.3, lambda ix: ix < 5), "Aluminio": (70000, 0.33, lambda ix: ix >= 5)}
    steps = [[(n, 3, -100.0 / len(tip)) for n in tip], [(nid(10, 2, 2), 2, 50.0), (midedge, 1, 20.0)]]
    _deck(path, "C3D10", qn, qe, mats, fixed, steps)


def _run(tmp_path, deck, **kw):
    pytest.importorskip("dolfinx")
    from run_from_inp import run_from_inp
    out = tmp_path / "out"
    results = run_from_inp(str(deck), output_dir=str(out), units="mm-N-MPa-t-s", solver="lu", **kw)
    return json.loads((out / "run_manifest.json").read_text(encoding="utf-8")), results


def _bound(ref: str) -> float:
    # .frd: cada componente com +-0.5 ulp; |U| herda ate sqrt(3)/2 ulp; a referencia
    # anotada foi arredondada de novo (+0.5 ulp). Mesmo criterio de test_inp_generico.
    ulp = 10.0 ** -len(ref.split(".")[1])
    return (3 ** 0.5 / 2 + 0.5) * ulp


# |U|max do CalculiX 2.23 (6 algarismos do .frd) e von Mises da media dos pontos de
# integracao (.dat, 7 algarismos; anotado com 6).
CCX = {
    "simple_beam_c3d10": {"umax": ["1.57556"], "vm": ["38232.5"]},
    "beam_c3d10_curvo": {"umax": ["0.236961", "0.270389"], "vm": ["41.3748", "61.9425"]},
}


@pytest.mark.parametrize("name", sorted(CCX))
def test_c3d10_bate_com_calculix(tmp_path, name):
    deck = tmp_path / f"{name}.inp"
    globals()[name](str(deck))
    man, results = _run(tmp_path, deck)
    assert man["element_types"] == ["C3D10"]
    ref = CCX[name]
    assert len(man["steps"]) == len(ref["umax"])
    for k, s in enumerate(man["steps"]):
        assert abs(s["u_max"] - float(ref["umax"][k])) <= _bound(ref["umax"][k])
        vm = float(max(results[k][1]["von_mises"]))
        # referencia com 6 algarismos (+-0.5 ulp) vinda de componentes com 7 (< 0.5 ulp no VM)
        ulp = 10.0 ** -len(ref["vm"][k].split(".")[1])
        assert abs(vm - float(ref["vm"][k])) <= ulp
        tot, app = s["reaction_totals"], s["applied_totals"]
        for c in "xyz":
            assert abs(tot["RF" + c] + app["F" + c]) < 1e-6 * max(1.0, abs(app["F" + c]))


def test_c3d10_patch_distorcido_reproduz_campo_linear(tmp_path):
    # Campo linear prescrito nos nos do contorno, vertice interno deslocado (arestas
    # retas, jacobiano constante por elemento): u linear exato e tensao constante.
    # Elemento curvo nao entra aqui: com a regra de 4 pontos do CalculiX o patch
    # test deixa de ser exato (o curvo e coberto pela comparacao com o CalculiX).
    pytest.importorskip("dolfinx")
    import numpy as np
    nodes, hexes, nid = _grid(2, 2, 2, 2.0, 2.0, 2.0)
    nodes[nid(1, 1, 1)] = (1.13, 0.91, 1.07)
    qn, qe = _quad(nodes, _tets(nodes, hexes))
    A = np.array([[1e-3, 2e-4, 0.0], [0.0, -5e-4, 3e-4], [1e-4, 0.0, 2e-3]])
    bnd = [n for n, (x, y, z) in qn.items() if min(x, y, z) < 1e-12 or max(x, y, z) > 2 - 1e-12]
    deck = tmp_path / "patch.inp"
    with open(deck, "w") as f:
        f.write("*NODE\n")
        for n, (x, y, z) in sorted(qn.items()):
            f.write(f"{n}, {x:.12g}, {y:.12g}, {z:.12g}\n")
        f.write("*ELEMENT, TYPE=C3D10, ELSET=Eall\n")
        for eid, (_, c) in enumerate(qe, 1):
            f.write(", ".join(map(str, [eid] + c)) + "\n")
        f.write("*MATERIAL, NAME=M\n*ELASTIC\n200000, 0.3\n*SOLID SECTION, ELSET=Eall, MATERIAL=M\n")
        f.write("*STEP\n*STATIC\n*BOUNDARY\n")
        for n in bnd:
            u = A @ np.array(qn[n])
            for d in range(3):
                f.write(f"{n}, {d + 1}, {d + 1}, {u[d]:.15g}\n")
        f.write("*NODE FILE\nU\n*END STEP\n")
    man, results = _run(tmp_path, deck)
    uh = results[0][0]
    X = uh.function_space.tabulate_dof_coordinates()
    U = uh.x.array.reshape(X.shape[0], 3)
    assert np.max(np.abs(U - X @ A.T)) < 1e-12
    vm = np.asarray(results[0][1]["von_mises"])
    assert np.ptp(vm) < 1e-8 * np.max(vm)


def test_no_solto_e_aceito_e_carga_nele_e_erro(tmp_path):
    # O CalculiX aceita nos que nao pertencem a nenhum elemento.
    pytest.importorskip("dolfinx")
    deck = tmp_path / "orfao.inp"
    simple_beam_c3d10(str(deck), extra_nodes="99999, 5.0, 5.0, 5.0\n")
    man, _ = _run(tmp_path, deck)
    assert abs(man["steps"][0]["u_max"] - 1.57556) <= _bound("1.57556")
    txt = deck.read_text().replace("*NODE FILE", "*CLOAD\n99999, 3, -1.0\n*NODE FILE")
    deck2 = tmp_path / "orfao_carga.inp"
    deck2.write_text(txt)
    from fc_profile import FCProfileError
    with pytest.raises(FCProfileError, match="pertence a algum elemento"):
        _run(tmp_path / "b", deck2)


def test_c3d10_termico_bimaterial_analitico(tmp_path):
    # Conducao em serie, k=50 em x<50 e k=200 em x>50, T(0)=100, T(100)=20:
    # q = 80/(50/50 + 50/200) = 64; T(50) = 100 - 64 = 36 (exato em P2).
    pytest.importorskip("dolfinx")
    import numpy as np
    nodes, hexes, _ = _grid(4, 1, 1, 100.0, 10.0, 10.0)
    qn, qe = _quad(nodes, _tets(nodes, hexes))
    deck = tmp_path / "termico.inp"
    with open(deck, "w") as f:
        f.write("*NODE\n")
        for n, (x, y, z) in sorted(qn.items()):
            f.write(f"{n}, {x:.10g}, {y:.10g}, {z:.10g}\n")
        f.write("*ELEMENT, TYPE=C3D10, ELSET=Eall\n")
        for eid, (_, c) in enumerate(qe, 1):
            f.write(", ".join(map(str, [eid] + c)) + "\n")
        for nm, k, sel in (("A", 50.0, lambda ix: ix < 2), ("B", 200.0, lambda ix: ix >= 2)):
            ids = [str(e) for e, (ix, _) in enumerate(qe, 1) if sel(ix)]
            f.write(f"*ELSET, ELSET=E{nm}\n" + "\n".join(ids) + "\n")
            f.write(f"*MATERIAL, NAME={nm}\n*CONDUCTIVITY\n{k}\n*SOLID SECTION, ELSET=E{nm}, MATERIAL={nm}\n")
        f.write("*STEP\n*HEAT TRANSFER, STEADY STATE\n*BOUNDARY\n")
        for n, (x, _, _) in qn.items():
            if abs(x) < 1e-12:
                f.write(f"{n}, 11, 11, 100.0\n")
            elif abs(x - 100.0) < 1e-12:
                f.write(f"{n}, 11, 11, 20.0\n")
        f.write("*NODE FILE\nNT\n*END STEP\n")
    man, results = _run(tmp_path, deck)
    th = results[0][0]
    X = th.function_space.tabulate_dof_coordinates()
    T = th.x.array
    exact = np.where(X[:, 0] <= 50.0, 100.0 - 64.0 * X[:, 0] / 50.0, 36.0 - 16.0 * (X[:, 0] - 50.0) / 50.0)
    assert np.max(np.abs(T - exact)) < 1e-10
