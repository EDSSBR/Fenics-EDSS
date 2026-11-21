"""Caminho unico para .inp generico do CalculiX (decisao Valdir 17/11/2026).

Cobre: capacidades (erro explicito do que nao e implementado), restricoes da
configuracao, semantica de passos do CalculiX, materiais sem fallback e
escolha automatica do solver. As referencias numericas sao do CalculiX 2.23
(variantes dos exemplos rodadas no Nitro em 17/11/2026); a comparacao usa a
precisao com que o .frd imprime (6 algarismos), nao um criterio de aceite.
"""

from __future__ import annotations

import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from fc_profile import FCProfileError  # noqa: E402
from inp_capabilities import (  # noqa: E402
    InpNotImplementedError, InpRestrictedError, apply_restrictions,
    parse_restrictions, validate_capabilities,
)
from inp_parser import effective_steps, materials_by_element_strict, parse_inp  # noqa: E402
from job_io import parse_run_args  # noqa: E402

EXAMPLES = os.path.join(ROOT, "examples")


def example(filename):
    """Exemplo do repo na pasta numerada (examples/NNNN - Nome/<deck>.inp); exatamente um."""
    import glob
    found = glob.glob(os.path.join(EXAMPLES, "[0-9][0-9][0-9][0-9] -*", filename))
    assert len(found) == 1, (filename, found)
    return found[0]

CUBE = """*NODE
1, 0, 0, 0
2, 1, 0, 0
3, 1, 1, 0
4, 0, 1, 0
5, 0, 0, 1
6, 1, 0, 1
7, 1, 1, 1
8, 0, 1, 1
*ELEMENT, TYPE=C3D8, ELSET=Eall
1, 1, 2, 3, 4, 5, 6, 7, 8
*NSET, NSET=Base
1, 2, 3, 4
*MATERIAL, NAME=Aco
*ELASTIC
210000, 0.3
*SOLID SECTION, ELSET=Eall, MATERIAL=Aco
*BOUNDARY
Base, 1, 3
"""


def _w(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def _steps(tmp_path, text):
    return effective_steps(parse_inp(_w(tmp_path, "s.inp", CUBE + text)))


# --- capacidades -----------------------------------------------------------

def test_hyper_neo_hooke_e_aceito():
    # passo 7 (21/11/2026): *HYPERELASTIC, NEO HOOKE + NLGEOM implementados (tests/test_hyper.py)
    model = parse_inp(example("beam_hyper.inp"))
    validate_capabilities(model)


@pytest.mark.parametrize("card,frag", [
    ("*AMPLITUDE, NAME=A\n0, 0\n", "AMPLITUDE nao implementado"),
    ("*TIE, NAME=T\nA, B\n", "TIE nao implementado"),
    ("*CARTAO INVENTADO\n", "nao reconhecido"),
])
def test_cartao_nao_implementado_ou_desconhecido(tmp_path, card, frag):
    model = parse_inp(_w(tmp_path, "c.inp", CUBE + card + "*STEP\n*STATIC\n*END STEP\n"))
    with pytest.raises(InpNotImplementedError, match=frag):
        validate_capabilities(model)


def test_elemento_nao_implementado(tmp_path):
    text = CUBE.replace("TYPE=C3D8", "TYPE=C3D8R") + "*STEP\n*STATIC\n*END STEP\n"
    model = parse_inp(_w(tmp_path, "r.inp", text))
    with pytest.raises(InpNotImplementedError, match="C3D8R nao implementado"):
        validate_capabilities(model)


def test_heat_transiente_nao_implementado(tmp_path):
    model = parse_inp(_w(tmp_path, "h.inp", CUBE + "*STEP\n*HEAT TRANSFER\n0.1, 1\n*END STEP\n"))
    with pytest.raises(InpNotImplementedError, match="transiente"):
        validate_capabilities(model)


def test_cload_fora_do_step_e_erro(tmp_path):
    with pytest.raises(FCProfileError, match="fora de \\*STEP"):
        parse_inp(_w(tmp_path, "f.inp", CUBE + "*CLOAD\n7, 3, -1\n*STEP\n*STATIC\n*END STEP\n"))


def test_nset_inexistente_e_erro(tmp_path):
    with pytest.raises(FCProfileError, match="NSET inexistente"):
        parse_inp(_w(tmp_path, "n.inp", CUBE + "*STEP\n*STATIC\n*CLOAD\nTopo, 3, -1\n*END STEP\n"))


def test_cload_aceita_nset_e_generate(tmp_path):
    text = CUBE + "*NSET, NSET=Topo, GENERATE\n5, 8\n*STEP\n*STATIC\n*CLOAD\nTopo, 3, -1\n*END STEP\n"
    st = _steps(tmp_path, text)
    assert sorted(cl.node for cl in st[0].loads) == [5, 6, 7, 8]


def test_elemento_em_linhas_de_continuacao(tmp_path):
    text = CUBE.replace("1, 1, 2, 3, 4, 5, 6, 7, 8\n", "1, 1, 2, 3, 4,\n5, 6, 7, 8\n")
    model = parse_inp(_w(tmp_path, "cont.inp", text + "*STEP\n*STATIC\n*END STEP\n"))
    assert model.elements[1].nodes == [1, 2, 3, 4, 5, 6, 7, 8]


# --- materiais sem fallback ------------------------------------------------

def test_elemento_sem_secao_e_erro(tmp_path):
    text = CUBE.replace("*SOLID SECTION, ELSET=Eall, MATERIAL=Aco\n", "")
    model = parse_inp(_w(tmp_path, "m.inp", text + "*STEP\n*STATIC\n*END STEP\n"))
    with pytest.raises(FCProfileError, match="sem \\*SOLID SECTION"):
        materials_by_element_strict(model, "elastic")


def test_secao_com_material_inexistente_e_erro(tmp_path):
    text = CUBE.replace("MATERIAL=Aco", "MATERIAL=Titanio")
    model = parse_inp(_w(tmp_path, "m2.inp", text + "*STEP\n*STATIC\n*END STEP\n"))
    with pytest.raises(FCProfileError, match="MATERIAL inexistente"):
        materials_by_element_strict(model, "elastic")


def test_termico_exige_conductivity(tmp_path):
    model = parse_inp(_w(tmp_path, "k.inp", CUBE + "*STEP\n*HEAT TRANSFER, STEADY STATE\n*END STEP\n"))
    with pytest.raises(FCProfileError, match="sem \\*CONDUCTIVITY"):
        materials_by_element_strict(model, "conductivity")


# --- semantica de passos (medida no ccx 2.23) -------------------------------

def test_cload_repetido_no_passo_soma_e_passo_novo_substitui(tmp_path):
    text = ("*STEP\n*STATIC\n*CLOAD\n7, 3, -30\n7, 3, -30\n*END STEP\n"
            "*STEP\n*STATIC\n*CLOAD\n7, 3, -30\n8, 3, -5\n*END STEP\n"
            "*STEP\n*STATIC\n*CLOAD, OP=NEW\n6, 3, -1\n*END STEP\n")
    s1, s2, s3 = _steps(tmp_path, text)
    assert {(c.node, c.dof): c.value for c in s1.loads} == {(7, 3): -60.0}
    assert {(c.node, c.dof): c.value for c in s2.loads} == {(7, 3): -30.0, (8, 3): -5.0}
    assert {(c.node, c.dof): c.value for c in s3.loads} == {(6, 3): -1.0}


def test_boundary_do_modelo_persiste_e_op_new_remove(tmp_path):
    text = ("*STEP\n*STATIC\n*BOUNDARY\n7, 3, 3, -0.05\n*END STEP\n"
            "*STEP\n*STATIC\n*END STEP\n"
            "*STEP\n*STATIC\n*BOUNDARY, OP=NEW\n1, 1, 3\n*END STEP\n")
    s1, s2, s3 = _steps(tmp_path, text)
    bc1 = {(b.node, b.dof): b.value for b in s1.boundary}
    assert len(bc1) == 13 and bc1[(7, 3)] == -0.05
    assert {(b.node, b.dof) for b in s2.boundary} == set(bc1)
    assert {(b.node, b.dof) for b in s3.boundary} == {(1, 1), (1, 2), (1, 3)}


def test_dois_passos_mecanico_e_termico_nao_implementado(tmp_path):
    text = CUBE + "*STEP\n*STATIC\n*END STEP\n*STEP\n*HEAT TRANSFER, STEADY STATE\n*END STEP\n"
    with pytest.raises(InpNotImplementedError, match="acoplamento"):
        validate_capabilities(parse_inp(_w(tmp_path, "mix.inp", text)))


# --- restricoes da configuracao ---------------------------------------------

def test_restricoes_da_configuracao(tmp_path):
    model = parse_inp(example("beam_multistep.inp"))
    apply_restrictions(model, {})
    with pytest.raises(InpRestrictedError, match="nao permitidos"):
        apply_restrictions(model, parse_restrictions('{"element_types": ["C3D4"]}'))
    with pytest.raises(InpRestrictedError, match="no maximo 1"):
        apply_restrictions(model, parse_restrictions('{"max_steps": 1}'))
    with pytest.raises(InpRestrictedError, match="CLOAD bloqueado"):
        apply_restrictions(model, parse_restrictions('{"blocked_keywords": ["*cload"]}'))
    with pytest.raises(ValueError, match="chave desconhecida"):
        parse_restrictions('{"max_passos": 1}')


def test_cli_solver_auto_e_restricoes():
    a = parse_run_args(["x.inp", "--solver", "auto", "--auto-lu-max-dofs", "50000",
                        "--restrictions", '{"max_steps": 3}'])
    assert a.solver == "auto" and a.auto_lu_max_dofs == 50000 and a.restrictions == {"max_steps": 3}
    with pytest.raises(ValueError, match="exige --auto-lu-max-dofs"):
        parse_run_args(["x.inp", "--solver", "auto"])
    with pytest.raises(ValueError, match="so vale com --solver auto"):
        parse_run_args(["x.inp", "--solver", "lu", "--auto-lu-max-dofs", "10"])
    with pytest.raises(ValueError, match="rtol"):
        parse_run_args(["x.inp", "--solver", "auto", "--auto-lu-max-dofs", "10", "--rtol", "1e-6"])


# --- solver (DOLFINx) -------------------------------------------------------

def _run(tmp_path, deck, **kw):
    pytest.importorskip("dolfinx")
    from run_from_inp import run_from_inp
    out = tmp_path / "out"
    results = run_from_inp(deck, output_dir=str(out), units="mm-N-MPa-t-s", **kw)
    _run.last_results = results
    return json.loads((out / "run_manifest.json").read_text(encoding="utf-8")), out


# |U|max do CalculiX 2.23 (.frd, 6 algarismos) para os exemplos.
CCX_UMAX = {
    "beam_2elem": [0.34242],
    "beam_nset": [0.34242],
    "simple_beam": [0.65508],
    "beam_multimat": [0.44456],
    "beam_multistep": [0.117934, 0.41090],
}


@pytest.mark.parametrize("name", sorted(CCX_UMAX))
def test_exemplos_c3d8_batem_umax_do_ccx(tmp_path, name):
    man, _ = _run(tmp_path, example(name + ".inp"), solver="lu")
    got = [s["u_max"] for s in man["steps"]]
    ref = CCX_UMAX[name]
    assert len(got) == len(ref)
    for g, r in zip(got, ref):
        # Arredondamento do .frd, nao criterio de aceite: cada componente sai com
        # +-0.5 unidade da ultima casa (ulp); |U| herda ate sqrt(3)/2 ulp e o valor de
        # referencia anotado aqui foi arredondado de novo (+0.5 ulp).
        ulp = 10.0 ** -len(str(r).split(".")[1])
        assert abs(g - r) <= (3 ** 0.5 / 2 + 0.5) * ulp
    assert man["profile"] == "generico"
    for s in man["steps"]:
        tot = s["reaction_totals"]
        app = s["applied_totals"]
        assert abs(tot["RFz"] + app["Fz"]) < 1e-6 * max(1.0, abs(app["Fz"]))


def test_termico_bimaterial_serie_analitico(tmp_path):
    # Conducao 1D em serie: k=50 em x<0.5 e k=200 em x>0.5, T(0)=100, T(1)=0.
    # q = 100/(0.5/50 + 0.5/200) = 8000; T(0.5) = 100 - 8000*0.01 = 20 (exato em Q1).
    pytest.importorskip("dolfinx")
    man, out = _run(tmp_path, example("beam_thermal.inp"), solver="lu")
    s = man["steps"][0]
    assert s["analysis"] == "heat"
    assert abs(s["t_max"] - 100.0) < 1e-9 and abs(s["t_min"]) < 1e-9
    import numpy as np
    uh = _run.last_results[0][0]
    pts = uh.function_space.tabulate_dof_coordinates()
    mid = np.isclose(pts[:, 0], 0.5)
    assert mid.sum() == 4
    assert np.allclose(uh.x.array[mid], 20.0, atol=1e-9)
    assert list(out.glob("*.pvd"))


def test_solver_auto_escolhe_pelo_limite(tmp_path):
    deck = example("beam_2elem.inp")
    man, _ = _run(tmp_path, deck, solver="auto", auto_lu_max_dofs=1000)
    auto = man["steps"][0]["solver"]["auto"]
    assert auto["n_dofs"] == 36 and auto["chosen"] == "lu"
    man2, _ = _run(tmp_path / "b", deck, solver="auto", auto_lu_max_dofs=36)
    assert man2["steps"][0]["solver"]["auto"]["chosen"] == "cg-gamg"
    assert man2["steps"][0]["solver"]["type"] == "cg-gamg"
