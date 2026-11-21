#!/usr/bin/env python3
"""
Parser minimo de arquivo .inp (CalculiX/Abaqus) para FeniCS-EDSS.

Um unico caminho para qualquer .inp do CalculiX. O que o FenicsEDSS ainda
nao implementa e recusado com erro explicito (inp_capabilities.py); o
perfil FC (--profile fc) e so um conjunto extra de restricoes (fc_profile.py).
Semantica de passos igual a do CalculiX 2.23 (medida em 17/11/2026):
- *CLOAD repetido no mesmo no/dof dentro de um passo soma;
- num passo novo, o valor do no/dof informado substitui o anterior e os
  demais continuam ativos; OP=NEW remove todas as cargas anteriores;
- *BOUNDARY antes do primeiro *STEP vale para todos os passos; dentro de um
  passo, acrescenta/substitui; OP=NEW remove as condicoes anteriores.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from fc_profile import (
    EXPLICITLY_IGNORED_KEYWORDS,
    EXPLICITLY_IGNORED_PARAMS,
    FCParseRecord,
    FCProfileError,
    IgnoredItem,
    OutputRequest,
    keyword_params,
    normalize_keyword,
    validate_fc_cards,
    parse_report_payload,
    PARSE_NOT_SOLVED_BANNER,
)


@dataclass
class Node:
    id: int
    x: float
    y: float
    z: float


@dataclass
class Element:
    id: int
    type: str
    nodes: List[int]


@dataclass
class Material:
    name: str
    E: float = 0.0
    nu: float = 0.0
    k: float = 0.0  # condutividade térmica
    # Neo-Hooke lido de *HYPERELASTIC (mu, lambda). None = não informado.
    mu: Optional[float] = None
    lmbda: Optional[float] = None
    hyperelastic: bool = False
    has_elastic: bool = False
    has_conductivity: bool = False


@dataclass
class BoundaryCondition:
    node: int
    dof: int
    value: float


@dataclass
class ConcentratedLoad:
    node: int
    dof: int
    value: float


@dataclass
class SurfaceLoad:
    elset: str
    face: str
    value: float


@dataclass
class Step:
    loads: List[ConcentratedLoad] = field(default_factory=list)
    dloads: List[SurfaceLoad] = field(default_factory=list)
    boundary: List[BoundaryCondition] = field(default_factory=list)
    boundary_op_new: bool = False
    cload_op_new: bool = False
    dload_op_new: bool = False
    heat_steady: bool = False
    line: int = 0
    nlgeom: bool = False
    analysis_type: str = "static"
    static_increments: Optional[List[float]] = None
    step_params: List[str] = field(default_factory=list)
    start_time: float = 0.0


@dataclass
class INPModel:
    nodes: Dict[int, Node] = field(default_factory=dict)
    elements: Dict[int, Element] = field(default_factory=dict)
    materials: Dict[str, Material] = field(default_factory=dict)
    solid_sections: Dict[str, str] = field(default_factory=dict)
    elsets: Dict[str, List[int]] = field(default_factory=dict)
    nsets: Dict[str, List[int]] = field(default_factory=dict)
    boundary: List[BoundaryCondition] = field(default_factory=list)
    steps: List[Step] = field(default_factory=list)
    analysis_type: str = "static"
    nlgeom: bool = False
    hyperelastic: bool = False
    fc_record: Optional[FCParseRecord] = None
    element_nset_from_node_card: Optional[str] = None
    # *BOUNDARY antes do primeiro *STEP (vale para todos os passos).
    initial_boundary: List[BoundaryCondition] = field(default_factory=list)



def _is_keyword_line(raw: str) -> bool:
    s = raw.strip()
    return s.startswith("*") and not s.startswith("**")



def _resolve_set_aliases(sets, aliases):
    """Expande referencias a outros sets (ex.: ELSET Eall -> Evolumes)."""
    pending = {k: list(v) for k, v in aliases.items()}
    for _ in range(16):
        progressed = False
        for name, refs in list(pending.items()):
            still = []
            for ref in refs:
                hit = None
                for key in sets:
                    if key.upper() == ref.upper():
                        hit = key
                        break
                if hit is not None and (hit.upper() != name.upper()):
                    sets.setdefault(name, []).extend(sets.get(hit, []))
                    progressed = True
                else:
                    still.append(ref)
            pending[name] = still
        if not progressed:
            break


# Numero de nos por tipo de elemento (para ler linhas de continuacao).
NODES_PER_ELEMENT = {
    "C3D4": 4, "C3D6": 6, "C3D8": 8, "C3D8R": 8, "C3D8I": 8,
    "C3D10": 10, "C3D10T": 10, "C3D15": 15, "C3D20": 20, "C3D20R": 20,
}


def _op_new(line: str) -> bool:
    m = re.search(r"\bOP\s*=\s*(\w+)", line, flags=re.IGNORECASE)
    if not m:
        return False
    op = m.group(1).upper()
    if op not in ("NEW", "MOD"):
        raise FCProfileError("Parametro OP={0} invalido em: {1}".format(m.group(1), line.strip()))
    return op == "NEW"


def _set_targets(model, target: str, lineno: int, card: str) -> List[int]:
    """No (numero) ou nome de NSET -> lista de nos. NSET inexistente e erro."""
    if target.lstrip("+-").isdigit():
        return [int(target)]
    for nset_name, nodes in model.nsets.items():
        if nset_name.upper() == target.upper():
            return list(nodes)
    raise FCProfileError(
        "L{0}: *{1} referencia NSET inexistente: {2!r}".format(lineno, card, target)
    )


def _lookup_set(sets, ref: str, owner: str, lineno: int, card: str) -> List[int]:
    """Conteudo de um set ja definido, referenciado por nome dentro de outro set."""
    for key, vals in sets.items():
        if key.upper() == ref.upper():
            if key.upper() == owner.upper():
                raise FCProfileError("L{0}: *{1} {2} referencia a si mesmo".format(lineno, card, owner))
            return list(vals)
    raise FCProfileError(
        "L{0}: *{1} {2} referencia {3!r}, que nao e numero nem {1} definido antes".format(
            lineno, card, owner, ref
        )
    )


def _read_set_block(lines, i, generate):
    """Le os dados de *NSET/*ELSET. Devolve (ids, aliases, i)."""
    ids: List[int] = []
    aliases: List[str] = []
    while i < len(lines) and not _is_keyword_line(lines[i]):
        raw = lines[i].strip()
        if not raw or raw.startswith("**"):
            i += 1
            continue
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if generate:
            try:
                nums = [int(p) for p in parts]
            except ValueError as exc:
                raise FCProfileError(
                    "L{0}: GENERATE espera inicio, fim[, passo]: {1}".format(i + 1, raw)
                ) from exc
            if len(nums) not in (2, 3) or (len(nums) == 3 and nums[2] <= 0):
                raise FCProfileError(
                    "L{0}: GENERATE espera inicio, fim[, passo]: {1}".format(i + 1, raw)
                )
            step = nums[2] if len(nums) == 3 else 1
            ids.extend(range(nums[0], nums[1] + 1, step))
        else:
            for tok in parts:
                try:
                    ids.append(int(tok))
                except ValueError:
                    aliases.append(tok)
        i += 1
    return ids, aliases, i


def parse_inp(path: str, debug: int = 0, profile: Optional[str] = None) -> INPModel:
    """Le um arquivo .inp e retorna um INPModel."""
    model = INPModel()
    current_material = None
    current_step = None
    fc_record = FCParseRecord()
    model.fc_record = fc_record
    elset_aliases = {}
    nset_aliases = {}
    profile_fc = (profile or "").strip().lower() == "fc"


    with open(path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line.startswith("**"):
            i += 1
            continue

        if not line.startswith("*"):
            i += 1
            continue

        key = normalize_keyword(line)
        lineno = i + 1
        fc_record.keywords.append((lineno, line))
        upper = line.upper()

        if key == "NODE":
            nset_m = re.search(r"NSET\s*=\s*(\w+)", line, flags=re.IGNORECASE)
            nset_name = nset_m.group(1) if nset_m else None
            i += 1
            while i < len(lines) and not _is_keyword_line(lines[i]):
                parts = [p.strip() for p in lines[i].split(",")]
                if len(parts) >= 4 and not lines[i].strip().startswith("**"):
                    nid = int(parts[0])
                    model.nodes[nid] = Node(nid, float(parts[1]), float(parts[2]), float(parts[3]))
                    if nset_name:
                        model.nsets.setdefault(nset_name, []).append(nid)
                i += 1
            continue

        if key == "ELEMENT":
            match = re.search(r"TYPE\s*=\s*(\w+)", line, flags=re.IGNORECASE)
            if match:
                etype = match.group(1).upper()
            else:
                raise FCProfileError(
                    "L{0}: *ELEMENT sem TYPE (o CalculiX exige TYPE; sem default C3D8).".format(lineno)
                )
            elset_m = re.search(r"ELSET\s*=\s*(\w+)", line, flags=re.IGNORECASE)
            elset_name = elset_m.group(1) if elset_m else None
            npe = NODES_PER_ELEMENT.get(etype)
            i += 1
            pending: List[int] = []
            while i < len(lines) and not _is_keyword_line(lines[i]):
                raw = lines[i].strip()
                if raw and not raw.startswith("**"):
                    pending.extend(int(p) for p in raw.split(",") if p.strip())
                    # Elemento completo: npe nos + id (ou linha unica se tipo desconhecido).
                    if npe is None or len(pending) >= npe + 1:
                        if npe is not None and len(pending) != npe + 1:
                            raise FCProfileError(
                                "L{0}: elemento {1} {2} com {3} nos (esperado {4}).".format(
                                    i + 1, pending[0], etype, len(pending) - 1, npe
                                )
                            )
                        if len(pending) >= 2:
                            eid = pending[0]
                            model.elements[eid] = Element(eid, etype, pending[1:])
                            if elset_name:
                                model.elsets.setdefault(elset_name, []).append(eid)
                        pending = []
                i += 1
            if pending:
                raise FCProfileError(
                    "L{0}: elemento {1} {2} incompleto ({3} nos).".format(
                        lineno, pending[0], etype, len(pending) - 1
                    )
                )
            continue

        if key == "MATERIAL":
            match = re.search(r"NAME\s*=\s*(\w+)", line, flags=re.IGNORECASE)
            if match:
                current_material = match.group(1)
                model.materials[current_material] = Material(name=current_material)
            i += 1
            continue

        if key == "ELASTIC" and current_material:
            i += 1
            while i < len(lines) and not _is_keyword_line(lines[i]):
                s = lines[i].strip()
                if s and not s.startswith("**"):
                    parts = [p.strip() for p in s.split(",")]
                    if len(parts) >= 2:
                        model.materials[current_material].E = float(parts[0])
                        model.materials[current_material].nu = float(parts[1])
                        model.materials[current_material].has_elastic = True
                    i += 1
                    break
                i += 1
            continue

        if key == "HYPERELASTIC" and current_material:
            mat = model.materials[current_material]
            mat.hyperelastic = True
            model.hyperelastic = True
            i += 1
            while i < len(lines) and not _is_keyword_line(lines[i]):
                data = lines[i].strip()
                if data and not data.startswith("**"):
                    parts = [p.strip() for p in data.split(",") if p.strip()]
                    if parts:
                        mat.mu = float(parts[0])
                    if len(parts) >= 2:
                        mat.lmbda = float(parts[1])
                    i += 1
                    break
                i += 1
            continue

        if key == "CONDUCTIVITY" and current_material:
            i += 1
            while i < len(lines) and not _is_keyword_line(lines[i]):
                s = lines[i].strip()
                if s and not s.startswith("**"):
                    parts = [p.strip() for p in s.split(",")]
                    if parts:
                        model.materials[current_material].k = float(parts[0])
                        model.materials[current_material].has_conductivity = True
                    i += 1
                    break
                i += 1
            continue

        if key == "SOLID SECTION":
            match_elset = re.search(r"ELSET\s*=\s*(\w+)", line, flags=re.IGNORECASE)
            match_mat = re.search(r"MATERIAL\s*=\s*(\w+)", line, flags=re.IGNORECASE)
            if match_elset and match_mat:
                model.solid_sections[match_elset.group(1)] = match_mat.group(1)
            i += 1
            continue

        if key == "NSET":
            match = re.search(r"NSET\s*=\s*(\w+)", line, flags=re.IGNORECASE)
            if match:
                nset_name = match.group(1)
                model.nsets.setdefault(nset_name, [])
                ids, al, i = _read_set_block(lines, i + 1, "GENERATE" in upper)
                model.nsets[nset_name].extend(ids)
                for ref in al:
                    model.nsets[nset_name].extend(_lookup_set(model.nsets, ref, nset_name, lineno, "NSET"))
                continue
            i += 1
            continue

        if key == "ELSET":
            match = re.search(r"ELSET\s*=\s*(\w+)", line, flags=re.IGNORECASE)
            if match:
                elset_name = match.group(1)
                model.elsets.setdefault(elset_name, [])
                ids, al, i = _read_set_block(lines, i + 1, "GENERATE" in upper)
                model.elsets[elset_name].extend(ids)
                for ref in al:
                    model.elsets[elset_name].extend(_lookup_set(model.elsets, ref, elset_name, lineno, "ELSET"))
                continue
            i += 1
            continue

        if key == "DLOAD":
            if current_step is None:
                raise FCProfileError(
                    "L{0}: *DLOAD fora de *STEP (o CalculiX exige cargas dentro do passo).".format(lineno)
                )
            if _op_new(line):
                current_step.dload_op_new = True
            i += 1
            while i < len(lines) and not _is_keyword_line(lines[i]):
                parts = [p.strip() for p in lines[i].split(",")]
                if len(parts) >= 3 and not lines[i].strip().startswith("**"):
                    elset = parts[0]
                    face = parts[1].upper()
                    try:
                        value = float(parts[2])
                    except ValueError as exc:
                        raise FCProfileError(
                            "L{0}: *DLOAD com valor invalido: {1}".format(i + 1, lines[i].strip())
                        ) from exc
                    current_step.dloads.append(SurfaceLoad(elset, face, value))
                i += 1
            continue

        if key == "NLGEOM":
            model.nlgeom = True
            i += 1
            continue

        if key == "BOUNDARY":
            bc_target_list = current_step.boundary if current_step is not None else model.initial_boundary
            if current_step is not None and _op_new(line):
                current_step.boundary_op_new = True
            i += 1
            while i < len(lines) and not _is_keyword_line(lines[i]):
                raw_parts = [p.strip() for p in lines[i].split(",")]
                while raw_parts and raw_parts[-1] == "":
                    raw_parts.pop()
                parts = raw_parts
                if len(parts) >= 2 and not lines[i].strip().startswith("**"):
                    target = parts[0]
                    if len(parts) == 3 and parts[2] != "":
                        third = parts[2]
                        looks_value = ("." in third) or ("e" in third.lower()) or (
                            third.lstrip("+-").isdigit() and int(third) not in range(1, 7)
                        )
                        if looks_value:
                            raise FCProfileError(
                                "L{0}: *BOUNDARY com 3 colunas e ambiguo. "
                                "CalculiX usa no, dof_ini, dof_fim[, valor]. "
                                "Use 2 colunas (valor 0) ou 4 colunas (no, ini, fim, valor). "
                                "Linha: {1}".format(i + 1, lines[i].strip())
                            )
                    try:
                        dof_start = int(parts[1])
                        if len(parts) >= 3 and parts[2] != "":
                            dof_end = int(parts[2])
                        else:
                            dof_end = dof_start
                        value = float(parts[3]) if len(parts) > 3 and parts[3] else 0.0
                    except ValueError as exc:
                        raise FCProfileError(
                            "L{0}: *BOUNDARY invalido no perfil FC ({1}). "
                            "Formato: no, dof_ini[, dof_fim[, valor]]. Linha: {2}".format(
                                i + 1, exc, lines[i].strip()
                            )
                        ) from exc
                    nodes_to_apply = _set_targets(model, target, i + 1, "BOUNDARY")
                    for nid in nodes_to_apply:
                        for dof in range(dof_start, dof_end + 1):
                            bc = BoundaryCondition(nid, dof, value)
                            model.boundary.append(bc)
                            bc_target_list.append(bc)
                i += 1
            continue

        if key == "STEP":
            current_step = Step()
            current_step.line = lineno
            current_step.step_params = keyword_params(line)
            current_step.start_time = 0.0
            for param in current_step.step_params:
                pname = param.split("=", 1)[0].strip().upper()
                if (key, pname) in EXPLICITLY_IGNORED_PARAMS:
                    fc_record.ignored.append(
                        IgnoredItem(
                            "STEP",
                            "parametro INC ignorado (perfil FC: um incremento, tempo inicial 0, sem heranca)",
                            lineno,
                            param,
                        )
                    )
            model.steps.append(current_step)
            fc_record.step_params = list(current_step.step_params)
            i += 1
            continue

        if key == "END STEP":
            current_step = None
            i += 1
            continue

        if key == "STATIC":
            if "NLGEOM" in upper:
                model.nlgeom = True
                if current_step:
                    current_step.nlgeom = True
            model.analysis_type = "static"
            if current_step:
                current_step.analysis_type = "static"
            i += 1
            while i < len(lines) and not _is_keyword_line(lines[i]):
                data = lines[i].strip()
                if data and not data.startswith("**"):
                    try:
                        inc = [float(x.strip()) for x in data.split(",") if x.strip()]
                    except ValueError:
                        inc = None
                    if inc:
                        if current_step is not None:
                            current_step.static_increments = inc
                        fc_record.static_increments = inc
                    i += 1
                    break
                i += 1
            continue

        if key == "HEAT TRANSFER":
            model.analysis_type = "heat"
            if current_step:
                current_step.analysis_type = "heat"
                current_step.heat_steady = "STEADY" in upper.replace(" ", "")
            i += 1
            continue

        if key == "CLOAD":
            if current_step is None:
                raise FCProfileError(
                    "L{0}: *CLOAD fora de *STEP (o CalculiX exige cargas dentro do passo).".format(lineno)
                )
            if _op_new(line):
                current_step.cload_op_new = True
            i += 1
            while i < len(lines) and not _is_keyword_line(lines[i]):
                parts = [p.strip() for p in lines[i].split(",")]
                if len(parts) >= 3 and not lines[i].strip().startswith("**"):
                    try:
                        dof = int(parts[1])
                        value = float(parts[2])
                    except ValueError as exc:
                        raise FCProfileError(
                            "L{0}: *CLOAD invalido (no|NSET, dof, valor): {1}".format(i + 1, lines[i].strip())
                        ) from exc
                    for node in _set_targets(model, parts[0], i + 1, "CLOAD"):
                        current_step.loads.append(ConcentratedLoad(node, dof, value))
                i += 1
            continue

        if key == "NODE FILE":
            req_line = lineno
            i += 1
            fields = []
            while i < len(lines) and not _is_keyword_line(lines[i]):
                s = lines[i].strip()
                if s and not s.startswith("**"):
                    fields.extend([x.strip().upper() for x in s.split(",") if x.strip()])
                i += 1
            fc_record.requests.append(OutputRequest("node_file", fields, line=req_line))
            continue

        if key == "EL FILE":
            req_line = lineno
            i += 1
            fields = []
            while i < len(lines) and not _is_keyword_line(lines[i]):
                s = lines[i].strip()
                if s and not s.startswith("**"):
                    fields.extend([x.strip().upper() for x in s.split(",") if x.strip()])
                i += 1
            fc_record.requests.append(OutputRequest("el_file", fields, line=req_line))
            continue

        if key == "NODE PRINT":
            req_line = lineno
            nset_m = re.search(r"NSET\s*=\s*(\w+)", line, flags=re.IGNORECASE)
            nset_name = nset_m.group(1) if nset_m else None
            totals = "TOTALS" in upper and "ONLY" in upper
            i += 1
            fields = []
            while i < len(lines) and not _is_keyword_line(lines[i]):
                s = lines[i].strip()
                if s and not s.startswith("**"):
                    fields.extend([x.strip().upper() for x in s.split(",") if x.strip()])
                i += 1
            fc_record.requests.append(
                OutputRequest("node_print", fields, nset=nset_name, totals_only=totals, line=req_line)
            )
            continue

        if key == "EL PRINT":
            req_line = lineno
            elset_m = re.search(r"ELSET\s*=\s*(\w+)", line, flags=re.IGNORECASE)
            elset_name = elset_m.group(1) if elset_m else None
            i += 1
            fields = []
            while i < len(lines) and not _is_keyword_line(lines[i]):
                s = lines[i].strip()
                if s and not s.startswith("**"):
                    fields.extend([x.strip().upper() for x in s.split(",") if x.strip()])
                i += 1
            fc_record.requests.append(
                OutputRequest("el_print", fields, elset=elset_name, line=req_line)
            )
            continue

        if key == "OUTPUT":
            # Parametros sao validados em validate_fc_cards: so FREQUENCY=1
            # e honrado. Reconhecer o cartao nao implica saida honrada.
            i += 1
            while i < len(lines) and not _is_keyword_line(lines[i]):
                s = lines[i].strip()
                if s and not s.startswith("**") and profile_fc:
                    raise FCProfileError(
                        "L{0}: dados sob *OUTPUT nao sao honrados no perfil FC: {1}".format(
                            i + 1, s
                        )
                    )
                i += 1
            continue

        if key in EXPLICITLY_IGNORED_KEYWORDS:
            if key == "EXPANSION":
                reason = (
                    "EXPANSION ignorado somente no perfil estritamente mecanico, "
                    "sem temperatura: o coeficiente nao entra na formulacao. "
                    "Cartao termico concomitante e rejeitado."
                )
            elif key == "DENSITY":
                reason = (
                    "DENSITY ignorado no perfil FC de estatica linear: a massa "
                    "nao entra na formulacao (sem gravitacao, inercia ou "
                    "carga de corpo). O valor e registrado como metadado, "
                    "nao e usado no solve."
                )
            else:
                reason = (
                    "metadado ignorado explicitamente (nao altera a estatica "
                    "mecanica do perfil FC)"
                )
            fc_record.ignored.append(
                IgnoredItem(
                    key,
                    reason,
                    lineno,
                    line,
                )
            )
            i += 1
            while i < len(lines) and not _is_keyword_line(lines[i]):
                i += 1
            continue

        # Keyword nao despachada: avanca o cartao e os dados (sem interpretar).
        i += 1
        while i < len(lines) and not _is_keyword_line(lines[i]):
            i += 1
        continue

    _resolve_set_aliases(model.elsets, elset_aliases)
    _resolve_set_aliases(model.nsets, nset_aliases)

    if profile_fc:
        types = {el.type for el in model.elements.values()}
        validate_fc_cards(fc_record, types, len(model.steps))

    return model


def select_analysis(model: INPModel, step: Step) -> str:
    """Escolhe o solver do step: ``heat``, ``hyper`` ou ``linear``."""
    if step.analysis_type == "heat" or model.analysis_type == "heat":
        return "heat"
    if step.nlgeom or model.nlgeom or model.hyperelastic:
        return "hyper"
    return "linear"


def _material_by_element(model: INPModel) -> Dict[int, Material]:
    """Mapa eid -> material. Primeiro elset com *SOLID SECTION vence.

    Cache no modelo: `elem_id in list` por elemento era O(N²) e travava
    malhas grandes (elsets sao listas).
    """
    cached = getattr(model, "_material_by_eid", None)
    if cached is not None:
        return cached
    mapping: Dict[int, Material] = {}
    section_by_elset = {sname.upper(): mname for sname, mname in model.solid_sections.items()}
    mat_by_name = {name.upper(): mat for name, mat in model.materials.items()}
    for elset_name, elem_ids in model.elsets.items():
        mat_name = section_by_elset.get(elset_name.upper())
        if not mat_name:
            continue
        mat = mat_by_name.get(mat_name.upper())
        if mat is None:
            continue
        for eid in elem_ids:
            if eid not in mapping:
                mapping[int(eid)] = mat
    model._material_by_eid = mapping
    return mapping


def get_material_for_element(model: INPModel, elem_id: int) -> Optional[Material]:
    """Retorna o material do elemento via elset/solid section.

    Se o elemento não estiver em nenhum elset com material, devolve o
    primeiro material do modelo (ou ``None`` se não houver materiais).
    """
    mapping = _material_by_element(model)
    if elem_id in mapping:
        return mapping[elem_id]
    if model.materials:
        return next(iter(model.materials.values()))
    return None


def materials_by_element_strict(model: INPModel, need: str) -> Dict[int, Material]:
    """eid -> Material via *SOLID SECTION, sem fallback. ``need``: 'elastic' ou 'conductivity'.

    Erros (como o CalculiX): secao com ELSET ou MATERIAL inexistente, elemento
    sem secao, elemento em duas secoes, material sem a propriedade pedida.
    """
    errors: List[str] = []
    elsets = {name.upper(): ids for name, ids in model.elsets.items()}
    mats = {name.upper(): mat for name, mat in model.materials.items()}
    mapping: Dict[int, Material] = {}
    for sname, mname in model.solid_sections.items():
        ids = elsets.get(sname.upper())
        mat = mats.get(mname.upper())
        if ids is None:
            errors.append("*SOLID SECTION com ELSET inexistente: {0}".format(sname))
            continue
        if mat is None:
            errors.append("*SOLID SECTION {0} com MATERIAL inexistente: {1}".format(sname, mname))
            continue
        for eid in ids:
            eid = int(eid)
            prev = mapping.get(eid)
            if prev is not None and prev is not mat:
                errors.append("Elemento {0} em duas *SOLID SECTION ({1} e {2})".format(eid, prev.name, mat.name))
                if len(errors) > 20:
                    break
            mapping[eid] = mat
    missing = [eid for eid in model.elements if eid not in mapping]
    if missing:
        errors.append("{0} elemento(s) sem *SOLID SECTION (ex.: {1}); sem material padrao".format(
            len(missing), missing[:5]))
    used = {id(m): m for m in mapping.values()}.values()
    for mat in used:
        if need == "elastic":
            if not mat.has_elastic:
                errors.append("Material {0} sem *ELASTIC".format(mat.name))
            elif not (mat.E > 0) or not (-1.0 < mat.nu < 0.5):
                errors.append("Material {0} com *ELASTIC invalido (E={1}, nu={2})".format(mat.name, mat.E, mat.nu))
        elif need == "conductivity":
            if not mat.has_conductivity:
                errors.append("Material {0} sem *CONDUCTIVITY".format(mat.name))
            elif not (mat.k > 0):
                errors.append("Material {0} com *CONDUCTIVITY invalida ({1})".format(mat.name, mat.k))
    if errors:
        raise FCProfileError("Materiais/secoes invalidos:\n- " + "\n- ".join(errors[:25]))
    return mapping


def effective_steps(model: INPModel) -> List[Step]:
    """Estado de cargas e condicoes de contorno ativo em cada passo (semantica CalculiX).

    Medido no ccx 2.23 (17/11/2026): dentro de um passo, *CLOAD repetido no mesmo
    no/dof soma; num passo novo, o no/dof informado substitui o valor anterior e os
    outros continuam; OP=NEW zera as cargas anteriores. *BOUNDARY: ultimo valor
    vence; OP=NEW remove as condicoes anteriores (inclusive as do modelo).
    *DLOAD segue a regra do *CLOAD por (elemento, face), medido no ccx 2.23 em
    21/11/2026: a mesma face repetida no passo soma (inclusive via ELSET e via id
    do elemento); num passo novo a face informada substitui e as outras continuam;
    OP=NEW zera. O estado efetivo guarda um SurfaceLoad por elemento (elset = id).
    """
    elsets_up = {name.upper(): ids for name, ids in model.elsets.items()}

    def _dload_elements(ref: str) -> List[int]:
        txt = ref.strip()
        if txt.lstrip("-").isdigit():
            eid = int(txt)
            if eid not in model.elements:
                raise FCProfileError("*DLOAD referencia elemento inexistente: {0}".format(eid))
            return [eid]
        ids = elsets_up.get(txt.upper())
        if ids is None:
            raise FCProfileError("*DLOAD referencia ELSET inexistente: {0}".format(ref))
        return [int(e) for e in ids]

    out: List[Step] = []
    bc_state: Dict[Tuple[int, int], float] = {}
    for bc in model.initial_boundary:
        bc_state[(bc.node, bc.dof)] = bc.value
    cl_state: Dict[Tuple[int, int], float] = {}
    dl_state: Dict[Tuple[int, str], float] = {}
    for step in model.steps:
        if step.boundary_op_new:
            bc_state = {}
        for bc in step.boundary:
            bc_state[(bc.node, bc.dof)] = bc.value
        if step.cload_op_new:
            cl_state = {}
        this_cl: Dict[Tuple[int, int], float] = {}
        for cl in step.loads:
            key = (cl.node, cl.dof)
            this_cl[key] = this_cl.get(key, 0.0) + cl.value
        cl_state.update(this_cl)
        if step.dload_op_new:
            dl_state = {}
        this_dl: Dict[Tuple[int, str], float] = {}
        for dl in step.dloads:
            for eid in _dload_elements(dl.elset):
                key = (eid, dl.face)
                this_dl[key] = this_dl.get(key, 0.0) + dl.value
        dl_state.update(this_dl)
        eff = Step(
            loads=[ConcentratedLoad(n, d, v) for (n, d), v in cl_state.items()],
            dloads=[SurfaceLoad(str(e), f, v) for (e, f), v in sorted(dl_state.items())],
            boundary=[BoundaryCondition(n, d, v) for (n, d), v in bc_state.items()],
            nlgeom=step.nlgeom,
            analysis_type=step.analysis_type,
            static_increments=step.static_increments,
            step_params=list(step.step_params),
            start_time=step.start_time,
            heat_steady=step.heat_steady,
            line=step.line,
        )
        out.append(eff)
    return out


def summary(model: INPModel) -> str:
    """Retorna um resumo do modelo lido."""
    lines = [
        f"Nós: {len(model.nodes)}",
        f"Elementos: {len(model.elements)}",
        f"Materiais: {list(model.materials.keys())}",
        f"Seções sólidas: {model.solid_sections}",
        f"Element sets: {list(model.elsets.keys())}",
        f"Node sets: {list(model.nsets.keys())}",
        f"Condições de contorno: {len(model.boundary)}",
        f"Steps: {len(model.steps)}",
        f"Tipo de análise: {model.analysis_type}",
        f"NLGEOM: {model.nlgeom}",
        f"Hiperelástico: {model.hyperelastic}",
    ]
    for i, step in enumerate(model.steps):
        lines.append(f"  Step {i+1}: {len(step.loads)} cargas, {len(step.dloads)} dloads, nlgeom={step.nlgeom}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    args = [a for a in sys.argv[1:] if a]
    profile = None
    if "--profile" in args:
        ix = args.index("--profile")
        if ix + 1 >= len(args):
            print("faltou o valor de --profile")
            sys.exit(2)
        profile = args[ix + 1]
        del args[ix:ix + 2]
    if args and args[0].startswith("--profile="):
        profile = args[0].split("=", 1)[1]
        args = args[1:]
    if len(args) < 1:
        print("Uso: python3 inp_parser.py arquivo.inp [--profile fc]")
        sys.exit(1)
    try:
        model = parse_inp(args[0], profile=profile)
    except FCProfileError as exc:
        print("ERRO FC: {0}".format(exc))
        sys.exit(2)
    print(summary(model))
    if (profile or "").strip().lower() == "fc":
        print(PARSE_NOT_SOLVED_BANNER)
        rec = model.fc_record
        if rec:
            print("Keywords:", len(rec.keywords))
            print("Ignorados explicitos:", [(i.keyword, i.reason) for i in rec.ignored])
            print("Pedidos de saida:", [(r.kind, r.fields, r.nset) for r in rec.requests])
    elif model.fc_record:
        rec = model.fc_record
        print("Keywords:", len(rec.keywords))
        print("Ignorados explicitos:", [(i.keyword, i.reason) for i in rec.ignored])
        print("Pedidos de saida:", [(r.kind, r.fields, r.nset) for r in rec.requests])
