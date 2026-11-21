"""Hiperelastico Neo-Hooke + NLGEOM comparado com o CalculiX (passo 7, 21/11/2026).

Referencias do CalculiX 2.23 (condaforge-2.23) rodado no Nitro em 21/11/2026 com
estes mesmos decks. Criterio: precisao de impressao do .frd (6 algarismos) e do
.dat (7 algarismos), nao um criterio de aceite.

Convergencia: o ccx para o Newton com o criterio padrao (residuo 0.005 da forca
media). Nos decks Neo-Hooke o resultado padrao e o apertado coincidem na
impressao; na viga St. Venant-Kirchhoff o padrao difere ~2e-3 em |U| (35.4533 x
35.4512), entao as referencias abaixo vem do ccx com
"*CONTROLS, PARAMETERS=FIELD / 1e-10, 1e-10" (so na copia do ccx; mesma fisica).

Semantica medida no ccx 2.23: NLGEOM so vale no *STEP (no *STATIC e ignorado com
aviso); *HYPERELASTIC liga NLGEOM no modelo; NLGEOM continua nos passos seguintes;
passo NLGEOM depois de passo linear chega ao mesmo equilibrio.
"""

from __future__ import annotations

import json
import math
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
FIX = os.path.join(os.path.dirname(__file__), "fixtures", "hyper")
EXAMPLES = os.path.join(ROOT, "examples")

from fc_profile import FCProfileError  # noqa: E402
from inp_capabilities import InpNotImplementedError, nlgeom_by_step, validate_capabilities  # noqa: E402
from inp_parser import parse_inp  # noqa: E402


def example(filename):
    import glob
    found = glob.glob(os.path.join(EXAMPLES, "[0-9][0-9][0-9][0-9] -*", filename))
    assert len(found) == 1, (filename, found)
    return found[0]


def _run(tmp_path, deck, **kw):
    pytest.importorskip("dolfinx")
    from run_from_inp import run_from_inp
    out = tmp_path / ("out_" + os.path.splitext(os.path.basename(str(deck)))[0])
    results = run_from_inp(str(deck), output_dir=str(out), units="mm-N-MPa-t-s", solver="lu", **kw)
    return json.loads((out / "run_manifest.json").read_text(encoding="utf-8")), results


def _ulp(ref: str) -> float:
    """ulp do valor como o ccx imprime (ex. '4.56053E+00' ou '30.8482')."""
    if "E" in ref.upper():
        mant, exp = ref.upper().split("E")
        return 10.0 ** (int(exp) - len(mant.split(".")[1]))
    return 10.0 ** -len(ref.split(".")[1])


def _bound_umax(ref: str) -> float:
    # mesmo criterio de test_c3d10/test_inp_generico (|U| de componentes do .frd)
    return (3 ** 0.5 / 2 + 0.5) * _ulp(ref)


def _bound_vm(vm_ref: str, max_comp: float) -> float:
    # .dat: componentes com 7 algarismos (+-0.5 ulp cada). |d VM| <= sqrt(3/2)*|d sigma|_F
    # <= sqrt(3/2)*3*0.5 ulp ~ 1.84 ulp do maior componente do elemento; mais 0.5 ulp da
    # referencia anotada com 7 algarismos.
    ulp_c = 10.0 ** (math.floor(math.log10(max_comp)) - 6)
    return 1.84 * ulp_c + 0.5 * _ulp(vm_ref)


def _node_u(results, k, model, nid):
    import numpy as np
    uh = results[k][0]
    X = uh.function_space.tabulate_dof_coordinates()
    n = model.nodes[nid]
    j = int(np.argmin(np.linalg.norm(X - np.array([n.x, n.y, n.z]), axis=1)))
    return uh.x.array.reshape(X.shape[0], -1)[j]


# 0003 beam_hyper (deck do Luiz, sem alteracao): deslocamentos do .frd do ccx, todos os nos.
BEAM_HYPER_FRD = {
    1: ("0.00000E+00", "0.00000E+00", "0.00000E+00"),
    2: ("-8.43660E-04", "-1.42820E-04", "-2.34481E-03"),
    3: ("-8.36340E+00", "-1.06444E+00", "-2.51430E+01"),
    4: ("0.00000E+00", "0.00000E+00", "0.00000E+00"),
    5: ("-8.43660E-04", "1.42820E-04", "-2.34481E-03"),
    6: ("-8.36340E+00", "1.06444E+00", "-2.51430E+01"),
    7: ("0.00000E+00", "0.00000E+00", "0.00000E+00"),
    8: ("7.69551E-04", "1.11359E-04", "-2.30726E-03"),
    9: ("4.56053E+00", "1.67442E+00", "-3.04632E+01"),
    10: ("0.00000E+00", "0.00000E+00", "0.00000E+00"),
    11: ("7.69551E-04", "-1.11359E-04", "-2.30726E-03"),
    12: ("4.56053E+00", "-1.67442E+00", "-3.04632E+01"),
}


def test_0003_beam_hyper_bate_com_calculix_no_a_no(tmp_path):
    deck = example("beam_hyper.inp")
    man, results = _run(tmp_path, deck)
    model = parse_inp(deck)
    assert man["steps"][0]["analysis"] == "static-nlgeom"
    for nid, ref in BEAM_HYPER_FRD.items():
        u = _node_u(results, 0, model, nid)
        for c in range(3):
            # componente do .frd: +-0.5 ulp
            assert abs(u[c] - float(ref[c])) <= 0.5 * _ulp(ref[c]), (nid, c, u[c], ref[c])
    assert abs(man["steps"][0]["u_max"] - 30.8482) <= _bound_umax("30.8482")
    # von Mises (media dos pontos de integracao, .dat): elemento 2, 0.9715000; maior |S| 2.07
    vm = float(max(results[0][1]["von_mises"]))
    assert abs(vm - 0.9715000) <= _bound_vm("0.9715000", 2.07)
    tot = man["steps"][0]["reaction_totals"]
    assert abs(tot["RFz"] - 200.0) <= 1e-9 * 200 and abs(tot["RFx"]) < 1e-6 and abs(tot["RFy"]) < 1e-6
    # o NLGEOM do *STATIC e ignorado como no ccx (registrado no manifesto)
    assert any(it["keyword"] == "STATIC" and "NLGEOM" in it["reason"] for it in man["ignored_cards"])


# deck: [(|U|max .frd, VM .dat, maior |S| do elemento, RF total do *NODE PRINT do NSET Fixos)]
CCX = {
    "bloco_nh_c3d8": [("2.25163", "0.4845287", 1.001, (-17.77778, 0, 0)),
                      ("1.26531", "0.2883667", 0.6424, (-11.11111, -3.0, 0))],
    "bloco_nh_c3d4": [("2.07074", "0.6352662", 1.016, (-17.77778, 0, 0)),
                      ("1.17283", "0.3706051", 0.5961, (-11.11111, -3.0, 0))],
    "bloco_nh_c3d10": [("2.70879", "0.7145692", 1.339, (-25.6, 0, 0))],
    "viga_svk_c3d8": [("35.4512", "175.6744", 3774.0, (0, 0, -2000.0))],
    "viga_svk_c3d4": [("13.6704", "2732.825", 3442.0, (0, 0, -2000.0))],
    # NLGEOM no passo 1 e nada no passo 2: continua NLGEOM (igual a NLGEOM nos dois)
    "svk_2p_nl_sem": [("35.4512", "175.6744", 3774.0, (0, 0, -2000.0)),
                      ("48.1845", "296.1268", 5938.0, (0, -150.0, -3000.0))],
    # passo 1 linear, passo 2 NLGEOM: chega ao mesmo equilibrio do caso acima
    "svk_2p_sem_nl": [("39.5014", "138.5642", 3024.0, (0, 0, -2000.0)),
                      ("48.1845", "296.1268", 5938.0, (0, -150.0, -3000.0))],
}


@pytest.mark.parametrize("name", sorted(CCX))
def test_nlgeom_bate_com_calculix(tmp_path, name):
    man, results = _run(tmp_path, os.path.join(FIX, name + ".inp"))
    refs = CCX[name]
    assert len(man["steps"]) == len(refs)
    for k, (s, (umax, vm_ref, max_comp, rf)) in enumerate(zip(man["steps"], refs)):
        assert abs(s["u_max"] - float(umax)) <= _bound_umax(umax), (k, s["u_max"], umax)
        vm = float(max(results[k][1]["von_mises"]))
        assert abs(vm - float(vm_ref)) <= _bound_vm(vm_ref, max_comp), (k, vm, vm_ref)
        sets = results[k][1]["reactions"]["node_print_totals"]["sets"]
        tot = sets["Fixos"]
        big = max(abs(v) for v in rf)
        ulp = 10.0 ** (math.floor(math.log10(big)) - 6)
        for c, v in zip("xyz", rf):
            assert abs(tot["RF" + c] - v) <= 0.5 * ulp + 1e-9 * big, (k, c, tot, rf)
    if name.startswith("svk_2p_sem"):
        assert [s["analysis"] for s in man["steps"]] == ["static", "static-nlgeom"]


def _deck(tmp_path, mat_card):
    text = open(os.path.join(FIX, "bloco_nh_c3d8.inp"), encoding="utf-8").read()
    assert "*HYPERELASTIC, NEO HOOKE\n0.5, 0.04\n" in text
    p = tmp_path / "d.inp"
    p.write_text(text.replace("*HYPERELASTIC, NEO HOOKE\n0.5, 0.04\n", mat_card), encoding="utf-8")
    return str(p)


@pytest.mark.parametrize("card,frag", [
    ("*HYPERELASTIC, MOONEY-RIVLIN\n0.5, 0.1, 0.04\n", "MOONEY-RIVLIN nao implementado"),
    ("*HYPERELASTIC, OGDEN, N=1\n1.0, 2.0, 0.04\n", "OGDEN, N nao implementado"),
    ("*HYPERELASTIC, POLYNOMIAL, N=2\n1, 2, 3, 4, 5, 6, 7\n", "POLYNOMIAL, N nao implementado"),
    ("*HYPERELASTIC, YEOH\n1, 2, 3, 0.1, 0, 0\n", "YEOH nao implementado"),
    ("*HYPERELASTIC, ARRUDA-BOYCE\n1, 2, 0.1\n", "ARRUDA-BOYCE nao implementado"),
    ("*HYPERELASTIC\n0.5, 0.1, 0.04\n", "Mooney-Rivlin"),
    ("*HYPERELASTIC, NEO HOOKE\n0.5, 0.04, 20.\n0.6, 0.04, 100.\n", "dependente de temperatura"),
])
def test_outras_formas_hiperelasticas_nao_implementadas(tmp_path, card, frag):
    model = parse_inp(_deck(tmp_path, card))
    with pytest.raises(InpNotImplementedError, match=frag):
        validate_capabilities(model)


@pytest.mark.parametrize("card,frag", [
    ("*HYPERELASTIC, NEO HOOKE\n0.5, 0.0\n", "incompressivel"),
    ("*ELASTIC\n10, 0.3\n*HYPERELASTIC, NEO HOOKE\n0.5, 0.04\n", "ao mesmo tempo"),
])
def test_material_hiperelastico_invalido_e_erro(tmp_path, card, frag):
    with pytest.raises(FCProfileError, match=frag):
        _run(tmp_path, _deck(tmp_path, card))


def test_nlgeom_no_depois_de_nlgeom_e_dload_com_nlgeom_sao_erro(tmp_path):
    text = open(os.path.join(FIX, "svk_2p_nl_sem.inp"), encoding="utf-8").read()
    i = text.rindex("*STEP\n")
    p = tmp_path / "no.inp"
    p.write_text(text[:i] + "*STEP, NLGEOM=NO\n" + text[i + 6:], encoding="utf-8")
    with pytest.raises(InpNotImplementedError, match="NLGEOM=NO"):
        validate_capabilities(parse_inp(str(p)))
    p2 = tmp_path / "dl.inp"
    p2.write_text(text.replace("*CLOAD\n", "*DLOAD\n1, P2, 1.0\n*CLOAD\n", 1), encoding="utf-8")
    with pytest.raises(InpNotImplementedError, match="DLOAD com NLGEOM"):
        validate_capabilities(parse_inp(str(p2)))


def test_semantica_nlgeom_como_o_calculix(tmp_path):
    text = open(os.path.join(FIX, "svk_2p_nl_sem.inp"), encoding="utf-8").read()
    # NLGEOM no *STEP do passo 1 continua no passo 2
    assert nlgeom_by_step(parse_inp(os.path.join(FIX, "svk_2p_nl_sem.inp"))) == [True, True]
    assert nlgeom_by_step(parse_inp(os.path.join(FIX, "svk_2p_sem_nl.inp"))) == [False, True]
    # NLGEOM no *STATIC: o ccx ignora (aviso) -> linear, registrado como ignorado
    p = tmp_path / "st.inp"
    p.write_text(text.replace("*STEP, NLGEOM\n*STATIC\n", "*STEP\n*STATIC, NLGEOM\n"), encoding="utf-8")
    m = parse_inp(str(p))
    assert nlgeom_by_step(m) == [False, False]
    assert any(it.keyword == "STATIC" and "NLGEOM" in it.reason for it in m.fc_record.ignored)
    # *HYPERELASTIC liga NLGEOM em todos os passos
    assert nlgeom_by_step(parse_inp(example("beam_hyper.inp"))) == [True]
