"""O que o FenicsEDSS implementa de um .inp do CalculiX, e restricoes por config.

Dois conceitos separados:

* Capacidade (fixa no codigo): cartao, parametro ou tipo de elemento que o
  solver sabe calcular. Fora disso o job falha com erro explicito
  ("nao implementado"), nunca com aproximacao silenciosa. Nao ha chave de
  configuracao que libere o que nao esta implementado.
* Restricao (vem do arquivo de configuracao do job, via ``--restrictions``):
  politica que limita ainda mais o que e aceito (tipos de elemento, numero de
  passos, cartoes bloqueados). Sem restricoes, vale tudo o que e implementado.
"""

from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional

from fc_profile import FCProfileError, keyword_params, normalize_keyword


class InpNotImplementedError(FCProfileError):
    """Cartao ou recurso do .inp que o FenicsEDSS nao implementa (exit 2)."""


class InpRestrictedError(FCProfileError):
    """Recurso implementado, mas bloqueado pelas restricoes da configuracao (exit 2)."""


# Tipos de elemento implementados (Lagrange P1/Q1/P2 do DOLFINx).
IMPLEMENTED_ELEMENTS = {"C3D4", "C3D8", "C3D10"}

# Cartoes interpretados pelo solver.
IMPLEMENTED_KEYWORDS = {
    "NODE", "ELEMENT", "NSET", "ELSET", "MATERIAL", "ELASTIC", "CONDUCTIVITY",
    "SOLID SECTION", "BOUNDARY", "CLOAD", "DLOAD", "STEP", "STATIC",
    "HEAT TRANSFER", "END STEP", "NODE FILE", "EL FILE", "NODE PRINT",
    "EL PRINT", "OUTPUT",
}

# Cartoes lidos e ignorados com motivo registrado no manifesto (nao mudam o
# resultado do que e implementado). EXPANSION so e ignorado sem fisica termica
# acoplada (verificado abaixo).
IGNORED_KEYWORDS = {
    "HEADING": "titulo, sem efeito no calculo",
    "PHYSICAL CONSTANTS": "constantes fisicas nao usadas em estatica linear/termica estacionaria",
    "DENSITY": "massa nao entra em estatica linear sem carga de corpo (GRAV/CENTRIF nao implementados)",
    "EXPANSION": "sem temperatura no passo mecanico, a dilatacao nao entra na formulacao",
}

# Cartoes conhecidos do CalculiX que ainda nao estao implementados.
NOT_IMPLEMENTED_KEYWORDS = {
    "HYPERELASTIC", "PLASTIC", "CREEP", "VISCO", "USER MATERIAL", "DEFORMATION PLASTICITY",
    "NLGEOM", "FREQUENCY", "BUCKLE", "DYNAMIC", "MODAL DYNAMIC", "STEADY STATE DYNAMICS",
    "COMPLEX FREQUENCY", "COUPLED TEMPERATURE-DISPLACEMENT", "UNCOUPLED TEMPERATURE-DISPLACEMENT",
    "THERMAL-ELECTRICAL", "ELECTROMAGNETICS", "SENSITIVITY", "VISCO",
    "CONTACT PAIR", "SURFACE INTERACTION", "SURFACE BEHAVIOR", "FRICTION", "GAP", "TIE",
    "EQUATION", "MPC", "RIGID BODY", "COUPLING", "DISTRIBUTING", "KINEMATIC",
    "AMPLITUDE", "INCLUDE", "TEMPERATURE", "INITIAL CONDITIONS", "CFLUX", "DFLUX",
    "FILM", "RADIATE", "SPECIFIC HEAT", "ORIENTATION", "TRANSFORM",
    "SHELL SECTION", "BEAM SECTION", "MEMBRANE SECTION", "SPRING", "DASHPOT",
    "MASS", "SURFACE", "RESTART", "CHANGE MATERIAL", "MODEL CHANGE", "SUBMODEL",
    "PRE-TENSION SECTION", "DISTRIBUTION", "USER ELEMENT", "BOUNDARYF",
}

# Tipos de *DLOAD implementados: pressao uniforme nas faces P1..P6.
DLOAD_PRESSURE_FACES = {"P1", "P2", "P3", "P4", "P5", "P6"}

RESTRICTION_KEYS = ("element_types", "max_steps", "blocked_keywords")


def parse_restrictions(raw: Optional[str]) -> Dict:
    """JSON da CLI (``--restrictions``) -> dict validado. Chave desconhecida e erro."""
    if raw is None or not str(raw).strip():
        return {}
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise ValueError("--restrictions nao e JSON valido: {0}".format(exc)) from exc
    if not isinstance(data, dict):
        raise ValueError("--restrictions deve ser um objeto JSON")
    unknown = sorted(set(data) - set(RESTRICTION_KEYS))
    if unknown:
        raise ValueError("--restrictions com chave desconhecida: {0} (validas: {1})".format(
            unknown, list(RESTRICTION_KEYS)))
    out: Dict = {}
    if data.get("element_types") is not None:
        vals = data["element_types"]
        if not isinstance(vals, list) or not all(isinstance(v, str) and v.strip() for v in vals):
            raise ValueError("restrictions.element_types deve ser lista de nomes (ex.: ['C3D4'])")
        out["element_types"] = sorted({v.strip().upper() for v in vals})
    if data.get("max_steps") is not None:
        val = data["max_steps"]
        if isinstance(val, bool) or not isinstance(val, int) or val < 1:
            raise ValueError("restrictions.max_steps deve ser inteiro >= 1")
        out["max_steps"] = val
    if data.get("blocked_keywords") is not None:
        vals = data["blocked_keywords"]
        if not isinstance(vals, list) or not all(isinstance(v, str) and v.strip() for v in vals):
            raise ValueError("restrictions.blocked_keywords deve ser lista de cartoes (ex.: ['DLOAD'])")
        out["blocked_keywords"] = sorted({v.strip().lstrip("*").upper() for v in vals})
    return out


def _has_param(header: str, name: str) -> bool:
    return any(p.split("=", 1)[0].strip().upper() == name for p in keyword_params(header))


def validate_capabilities(model) -> None:
    """Recusa (erro explicito) o que o FenicsEDSS nao implementa. Junta todos os motivos."""
    rec = model.fc_record
    errors: List[str] = []
    keys_seen = []
    for lineno, header in rec.keywords:
        key = normalize_keyword(header)
        keys_seen.append(key)
        if key in IMPLEMENTED_KEYWORDS or key in IGNORED_KEYWORDS:
            pass
        elif key in NOT_IMPLEMENTED_KEYWORDS:
            errors.append("L{0}: *{1} nao implementado no FenicsEDSS. Cartao: {2}".format(
                lineno, key, header.strip()))
            continue
        else:
            errors.append("L{0}: cartao nao reconhecido pelo FenicsEDSS: {1}".format(lineno, header.strip()))
            continue
        up = header.upper().replace(" ", "")
        if key in ("STATIC", "STEP") and _has_param(header, "NLGEOM"):
            errors.append("L{0}: NLGEOM (nao linearidade geometrica) nao implementado. Cartao: {1}".format(
                lineno, header.strip()))
        if key == "STEP" and _has_param(header, "PERTURBATION"):
            errors.append("L{0}: *STEP, PERTURBATION nao implementado.".format(lineno))
        if key == "HEAT TRANSFER" and "STEADYSTATE" not in up:
            errors.append(
                "L{0}: *HEAT TRANSFER transiente nao implementado (use STEADY STATE). Cartao: {1}".format(
                    lineno, header.strip()))
        if key == "ELASTIC":
            for p in keyword_params(header):
                name, _, val = p.partition("=")
                if name.strip().upper() == "TYPE" and val.strip().upper() not in ("ISO", "ISOTROPIC"):
                    errors.append("L{0}: *ELASTIC, TYPE={1} nao implementado (so isotropico).".format(
                        lineno, val.strip()))
        if key == "BOUNDARY" and (_has_param(header, "AMPLITUDE") or _has_param(header, "TYPE")
                                  or _has_param(header, "FIXED") or _has_param(header, "SUBMODEL")):
            errors.append("L{0}: parametro de *BOUNDARY nao implementado: {1}".format(lineno, header.strip()))
        if key in ("CLOAD", "DLOAD") and (_has_param(header, "AMPLITUDE") or _has_param(header, "TIME DELAY")
                                          or _has_param(header, "LOAD CASE") or _has_param(header, "SECTOR")):
            errors.append("L{0}: parametro de *{1} nao implementado: {2}".format(lineno, key, header.strip()))

    if "EXPANSION" in keys_seen and any(k in ("TEMPERATURE", "INITIAL CONDITIONS", "HEAT TRANSFER") for k in keys_seen):
        errors.append("*EXPANSION com fisica termica no mesmo deck: acoplamento termo-mecanico nao implementado.")

    types = sorted({el.type.upper() for el in model.elements.values()})
    if not types:
        errors.append("O .inp nao tem *ELEMENT.")
    for t in types:
        if t not in IMPLEMENTED_ELEMENTS:
            errors.append("Elemento {0} nao implementado no FenicsEDSS (implementados: {1}).".format(
                t, sorted(IMPLEMENTED_ELEMENTS)))
    if len(types) > 1:
        errors.append("Malha com tipos de elemento misturados nao implementada: {0}.".format(types))

    kinds = {s.analysis_type for s in model.steps}
    if len(kinds) > 1:
        errors.append("Passos mecanicos e termicos no mesmo deck nao implementados (acoplamento).")
    if not model.steps:
        errors.append("O .inp nao tem *STEP (o CalculiX exige ao menos um passo).")
    for n, step in enumerate(model.steps, start=1):
        for dl in step.dloads:
            if dl.face not in DLOAD_PRESSURE_FACES:
                errors.append("Passo {0}: *DLOAD tipo {1} nao implementado (so pressao P1..P6).".format(
                    n, dl.face))
        if step.analysis_type == "heat" and (step.loads or step.dloads):
            errors.append("Passo {0}: carga mecanica em passo termico nao implementada.".format(n))

    if errors:
        raise InpNotImplementedError("O FenicsEDSS recusou o .inp:\n- " + "\n- ".join(errors))


def apply_restrictions(model, restrictions: Dict) -> None:
    """Aplica as restricoes da configuracao do job. Erro explicito se violadas."""
    if not restrictions:
        return
    errors: List[str] = []
    allowed = restrictions.get("element_types")
    if allowed:
        types = sorted({el.type.upper() for el in model.elements.values()})
        bad = [t for t in types if t not in allowed]
        if bad:
            errors.append("tipos de elemento {0} nao permitidos pela configuracao (permitidos: {1})".format(
                bad, allowed))
    max_steps = restrictions.get("max_steps")
    if max_steps is not None and len(model.steps) > max_steps:
        errors.append("{0} passos; a configuracao permite no maximo {1}".format(len(model.steps), max_steps))
    blocked = set(restrictions.get("blocked_keywords") or [])
    if blocked:
        for lineno, header in model.fc_record.keywords:
            key = normalize_keyword(header)
            if key in blocked:
                errors.append("L{0}: *{1} bloqueado pela configuracao".format(lineno, key))
    if errors:
        raise InpRestrictedError("Restricoes da configuracao violadas:\n- " + "\n- ".join(errors))


def ignored_items(model) -> List[Dict]:
    """Cartoes ignorados com motivo (vai para o manifesto)."""
    out = []
    for lineno, header in model.fc_record.keywords:
        key = normalize_keyword(header)
        if key in IGNORED_KEYWORDS:
            out.append({"keyword": key, "reason": IGNORED_KEYWORDS[key], "line": lineno, "detail": header.strip()})
    return out
