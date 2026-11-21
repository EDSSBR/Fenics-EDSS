"""*DLOAD (pressao em faces P1..P6) comparado com o CalculiX (passo 6, 21/11/2026).

Referencias do CalculiX 2.23 (condaforge-2.23) rodado no Nitro em 21/11/2026 com
estes mesmos decks. O criterio e a precisao de impressao do .frd/.dat, nao um
criterio de aceite. Os decks p_*.inp em fixtures/dload vieram de gerar_decks.py:
grade 10x2x2 de viga 100x10x10, aco/aluminio, pressao diferente em cada plano
externo (inclusive na face engastada), 3 passos (todas / troca duas + CLOAD /
OP=NEW com uma so). Os tetraedros usam permutacoes pares dos nos para que as
faces P1..P4 aparecam todas; o C3D10 curvo tem faces externas curvas e o C3D8
distorcido tem faces nao planas.

RF segue a definicao do CalculiX: K u nos DOFs prescritos; carga aplicada num DOF
prescrito vai direto ao apoio e nao entra no RF (medido no ccx 2.23).
"""

from __future__ import annotations

import json
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
FIX = os.path.join(os.path.dirname(__file__), "fixtures", "dload")

from fc_profile import FCProfileError  # noqa: E402
from inp_capabilities import InpNotImplementedError  # noqa: E402

# Viga de 2 C3D8 (0002 beam_dload do Luiz, com NSET para RF).
BEAM = """*NODE
1, 0.0, 0.0, 0.0
2, 0.5, 0.0, 0.0
3, 1.0, 0.0, 0.0
4, 0.0, 0.2, 0.0
5, 0.5, 0.2, 0.0
6, 1.0, 0.2, 0.0
7, 0.0, 0.0, 0.2
8, 0.5, 0.0, 0.2
9, 1.0, 0.0, 0.2
10, 0.0, 0.2, 0.2
11, 0.5, 0.2, 0.2
12, 1.0, 0.2, 0.2
*ELEMENT, TYPE=C3D8, ELSET=Eall
1, 1, 2, 5, 4, 7, 8, 11, 10
2, 2, 3, 6, 5, 8, 9, 12, 11
*ELSET, ELSET=Esquerda
1
*ELSET, ELSET=Direita
2
*MATERIAL, NAME=Aco
*ELASTIC
210000, 0.3
*MATERIAL, NAME=Aluminio
*ELASTIC
70000, 0.33
*SOLID SECTION, ELSET=Esquerda, MATERIAL=Aco
*SOLID SECTION, ELSET=Direita, MATERIAL=Aluminio
*NSET, NSET=Fixos
1, 4, 7, 10
*BOUNDARY
Fixos, 1, 3, 0.0
"""
OUT = "*NODE FILE\nU\n*NODE PRINT, NSET=Fixos, TOTALS=YES\nRF\n*END STEP\n"


def _beam(tmp_path, name, *steps):
    p = tmp_path / (name + ".inp")
    p.write_text(BEAM + "".join("*STEP\n*STATIC\n" + s + OUT for s in steps), encoding="utf-8")
    return p


def _run(tmp_path, deck, **kw):
    pytest.importorskip("dolfinx")
    from run_from_inp import run_from_inp
    out = tmp_path / ("out_" + os.path.splitext(os.path.basename(str(deck)))[0])
    results = run_from_inp(str(deck), output_dir=str(out), units="mm-N-MPa-t-s", solver="lu", **kw)
    return json.loads((out / "run_manifest.json").read_text(encoding="utf-8")), results


def _bound(ref: str) -> float:
    # mesmo criterio de test_c3d10/test_inp_generico (|U| de componentes do .frd)
    ulp = 10.0 ** -len(ref.split(".")[1])
    return (3 ** 0.5 / 2 + 0.5) * ulp


def _rf_ok(tot, ref):
    # .dat com 7 algarismos: meio ulp do maior componente do passo
    big = max(abs(v) for v in ref)
    ulp = 10.0 ** (len(str(int(big))) - 7) if big >= 1 else 1e-7
    for c, v in zip("xyz", ref):
        assert abs(tot["RF" + c] - v) <= 0.5 * ulp + 1e-9 * big, (c, tot, ref)


# |U|max (.frd), RF total do .dat por passo, CalculiX 2.23.
SEMANTICA = {
    "beam_dload": (["Direita, P1, -100.0\n"], ["0.0276587"], [(0, 0, 10.0)]),
    # mesma face repetida no passo SOMA (ELSET + id do elemento)
    "a_dup_mesmo_passo": (["Direita, P1, -100.0\n2, P1, -50.0\n"], ["0.0414880"], [(0, 0, 15.0)]),
    "a2_dup_mesmo_cartao_id": (["2, P1, -100.0\n2, P1, -50.0\n"], ["0.0414880"], [(0, 0, 15.0)]),
    # passo novo substitui a mesma face e mantem as outras
    "b_passo_novo_substitui": (["Direita, P1, -100.0\nDireita, P3, -10.0\n", "2, P1, -30.0\n"],
                               ["0.0278840", "0.00882942"], [(0, 1.0, 10.0), (0, 1.0, 3.0)]),
    # OP=NEW limpa
    "c_op_new": (["Direita, P1, -100.0\n", "OPNEW Direita, P3, -10.0\n"],
                 ["0.0276587", "0.00276587"], [(0, 0, 10.0), (0, 1.0, 0)]),
    # passo sem *DLOAD mantem as pressoes anteriores
    "d_sem_dload_no_passo2": (["Direita, P1, -100.0\n", None], ["0.0276587", "0.0284212"],
                              [(0, 0, 10.0), (-1.0, 0, 10.0)]),
}


def _sem_steps(dl):
    out = []
    for s in dl:
        if s is None:
            out.append("*CLOAD\n9, 1, 1.0\n")
        elif s.startswith("OPNEW "):
            out.append("*DLOAD, OP=NEW\n" + s[6:])
        else:
            out.append("*DLOAD\n" + s)
    return out


@pytest.mark.parametrize("name", sorted(SEMANTICA))
def test_dload_semantica_de_passos_bate_com_calculix(tmp_path, name):
    dl, umax, rf = SEMANTICA[name]
    man, _ = _run(tmp_path, _beam(tmp_path, name, *_sem_steps(dl)))
    assert len(man["steps"]) == len(umax)
    for s, u, r in zip(man["steps"], umax, rf):
        assert abs(s["u_max"] - float(u)) <= _bound(u), (s["u_max"], u)
        _rf_ok(s["reaction_totals"], r)


def test_beam_dload_von_mises(tmp_path):
    _man, results = _run(tmp_path, _beam(tmp_path, "vm", "*DLOAD\nDireita, P1, -100.0\n"))
    # CalculiX: 433.3090 (media dos pontos de integracao, .dat)
    assert abs(float(max(results[0][1]["von_mises"])) - 433.3090) <= 1e-4


# (|U|max por passo, VM max elemento por passo, RF por passo), CalculiX 2.23
GERADOS = {
    "p_c3d8": (["2.90449", "3.39093", "1.68859"], ["572.656", "649.510", "245.779"],
               [(300.0, 2375.0, -4275.0), (-200.0, 2350.0, 5225.0), (0, 2850.0, 0)]),
    "p_c3d8_distorcido": (["2.81260", "3.31934", "1.63410"], ["558.257", "670.544", "276.415"],
                          [(297.75, 2361.555, -4275.0), (-196.625, 2359.055, 5225.0), (0, 2851.38, 0)]),
    "p_c3d4": (["1.63953", "1.59657", "0.874436"], ["773.211", "662.500", "372.224"],
               [(300.0, 2375.0, -4275.0), (-200.0, 2350.0, 5225.0), (0, 2850.0, 0)]),
    "p_c3d10": (["4.12675", "4.81267", "2.39808"], ["1359.88", "1364.16", "583.550"],
                [(300.0, 2458.333, -4425.0), (-200.0, 2433.333, 5408.333), (0, 2950.0, 0)]),
    "p_c3d10_curvo": (["4.11756", "4.78668", "2.39411"], ["1344.29", "1344.62", "585.533"],
                      [(298.507, 2458.333, -4425.0), (-197.7605, 2433.333, 5408.333), (0, 2950.0, 0)]),
}


@pytest.mark.parametrize("name", sorted(GERADOS))
def test_dload_todas_as_faces_bate_com_calculix(tmp_path, name):
    umax, vms, rfs = GERADOS[name]
    man, results = _run(tmp_path, os.path.join(FIX, name + ".inp"))
    assert len(man["steps"]) == 3
    for k, s in enumerate(man["steps"]):
        assert abs(s["u_max"] - float(umax[k])) <= _bound(umax[k]), (k, s["u_max"], umax[k])
        vm = float(max(results[k][1]["von_mises"]))
        ulp = 10.0 ** -len(vms[k].split(".")[1])
        assert abs(vm - float(vms[k])) <= ulp, (k, vm, vms[k])
        _rf_ok(s["reaction_totals"], rfs[k])


@pytest.mark.parametrize(
    "linha,erro,frag",
    [
        ("1, P4, 1.0\n", InpNotImplementedError, "face interna"),
        ("Eall, GRAV, 9810., 0., 0., -1.\n", InpNotImplementedError, "GRAV nao implementado"),
        ("Eall, CENTRIF, 1.0, 0., 0., 0., 1., 0., 0.\n", InpNotImplementedError, "CENTRIF nao implementado"),
        ("2, P7, 1.0\n", InpNotImplementedError, "P7 nao implementado"),
        ("Nada, P1, 1.0\n", FCProfileError, "ELSET inexistente"),
        ("99, P1, 1.0\n", FCProfileError, "elemento inexistente"),
    ],
)
def test_dload_fora_do_escopo_e_erro_explicito(tmp_path, linha, erro, frag):
    with pytest.raises(erro, match=frag):
        _run(tmp_path, _beam(tmp_path, "bad", "*DLOAD\n" + linha))
