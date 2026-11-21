"""Decks de teste de *DLOAD (pressao) para C3D4, C3D8, C3D10 (reto e curvo) e C3D8 distorcido.
Pressoes diferentes por plano externo, cobrindo todas as faces P1..P6 / P1..P4 que aparecem;
3 passos: (1) todas; (2) muda duas e acrescenta CLOAD; (3) OP=NEW com uma so."""
import os
src = open("/home/valdir/mike-tmp/c3d10/gen_c3d10.py").read().split("# (a)")[0]
g = {}; exec(src, g)
OUT = "/home/valdir/mike-tmp/dload/c1/decks"; os.makedirs(OUT, exist_ok=True)
FACES = {"C3D8": {"P1": (0,1,2,3), "P2": (4,7,6,5), "P3": (0,4,5,1), "P4": (1,5,6,2), "P5": (2,6,7,3), "P6": (3,7,4,0)},
         "C3D4": {"P1": (0,1,2), "P2": (0,3,1), "P3": (1,3,2), "P4": (2,3,0)}}
FACES["C3D10"] = FACES["C3D4"]
L, B, H = 100.0, 10.0, 10.0
PLANES = {"x0": (0, 0.0, 2.0), "xL": (0, L, 3.0), "y0": (1, 0.0, -1.5), "yB": (1, B, 1.0), "z0": (2, 0.0, 0.5), "zH": (2, H, -4.0)}

def classify(etype, els, X0):
    """(eid, face) por plano, pelos nos de canto na geometria reta original X0."""
    out = {k: [] for k in PLANES}
    for eid, (_, c) in enumerate(els, 1):
        for fname, idx in FACES[etype].items():
            pts = [X0[c[i]] for i in idx]
            for k, (ax, val, _) in PLANES.items():
                if all(abs(p[ax] - val) < 1e-9 for p in pts): out[k].append((eid, fname))
    return out

def write(name, etype, nodes, els, X0):
    mats = {"Aco": (210000, 0.3, lambda ix: ix < 5), "Aluminio": (70000, 0.33, lambda ix: ix >= 5)}
    fixed = sorted(n for n, (x, y, z) in X0.items() if abs(x) < 1e-12 and n in {m for _, c in els for m in c})
    faces = classify(etype, els, X0)
    used = sorted({f for v in faces.values() for _, f in v})
    with open(f"{OUT}/{name}.inp", "w") as f:
        f.write("*NODE\n"); [f.write(f"{n}, {x:.10g}, {y:.10g}, {z:.10g}\n") for n, (x, y, z) in sorted(nodes.items())]
        f.write(f"*ELEMENT, TYPE={etype}, ELSET=Eall\n")
        for eid, (_, c) in enumerate(els, 1):
            s = [str(eid)] + [str(n) for n in c]
            f.write(", ".join(s[:16]) + ("\n" if len(s) <= 16 else ",\n" + ", ".join(s[16:]) + "\n"))
        for m, (E, nu, sel) in mats.items():
            ids = [str(e) for e, (ix, _) in enumerate(els, 1) if sel(ix)]
            f.write(f"*ELSET, ELSET=E_{m}\n"); [f.write(", ".join(ids[i:i+16]) + "\n") for i in range(0, len(ids), 16)]
            f.write(f"*MATERIAL, NAME={m}\n*ELASTIC\n{E}, {nu}\n*SOLID SECTION, ELSET=E_{m}, MATERIAL={m}\n")
        # ELSET com os elementos da face de topo (para testar DLOAD por ELSET); face e a mesma em todos? so no C3D8
        f.write("*NSET, NSET=MIKEFIX\n"); [f.write(", ".join(map(str, fixed[i:i+16])) + "\n") for i in range(0, len(fixed), 16)]
        f.write("*BOUNDARY\nMIKEFIX, 1, 3, 0.0\n")
        outreq = "*NODE FILE\nU\n*EL PRINT, ELSET=Eall\nS\n*NODE PRINT, NSET=MIKEFIX, TOTALS=YES\nRF\n*END STEP\n"
        f.write("*STEP\n*STATIC\n*DLOAD\n")
        for k, lst in faces.items():
            for eid, fn in lst: f.write(f"{eid}, {fn}, {PLANES[k][2]}\n")
        f.write(outreq)
        f.write("*STEP\n*STATIC\n*DLOAD\n")
        for eid, fn in faces["zH"]: f.write(f"{eid}, {fn}, 6.0\n")
        for eid, fn in faces["xL"]: f.write(f"{eid}, {fn}, -2.0\n")
        f.write("*CLOAD\n%d, 2, 25.0\n" % max(n for n, (x, y, z) in X0.items() if abs(x - L) < 1e-12 and n in nodes))
        f.write(outreq)
        f.write("*STEP\n*STATIC\n*CLOAD, OP=NEW\n*DLOAD, OP=NEW\n")
        for eid, fn in faces["yB"]: f.write(f"{eid}, {fn}, 3.0\n")
        f.write(outreq)
    print(name, etype, "faces usadas:", used, {k: len(v) for k, v in faces.items()})

nodes, hexes, nid = g["grid"](10, 2, 2, L, B, H)
write("p_c3d8", "C3D8", nodes, hexes, nodes)
t = g["tets"](nodes, hexes)
# permutacoes pares dos nos (orientacao mantida) para que todas as faces P1..P4 aparecam
PERM = [(0,1,2,3),(1,2,0,3),(0,2,3,1),(1,0,3,2)]
t = [(ix, [v[q] for q in PERM[k % 4]]) for k, (ix, v) in enumerate(t)]
write("p_c3d4", "C3D4", nodes, t, nodes)
qn, qe = g["quad"](nodes, t)
write("p_c3d10", "C3D10", qn, qe, qn)
# C3D10 curvo: nos de meio de aresta deslocados, inclusive nas faces externas (faces curvas)
qc = dict(qn)
for n in {m for _, c in qe for m in c[4:]}:
    x, y, z = qc[n]; s = 0.25 * ((n % 7) - 3) / 3.0
    if 0 < x < L: qc[n] = (x, y + (s if 0 < y < B else 0.0), z + (s if 0 < z < H else 0.0 if True else 0))
    # faces externas: deslocar no plano da face nao curva; deslocar normal curva a face
for n in {m for _, c in qe for m in c[4:]}:
    x, y, z = qc[n]
    if 0 < x < L and abs(z - H) < 1e-12 and 0 < y < B: qc[n] = (x, y, H + 0.2 * ((n % 5) - 2) / 2.0)
write("p_c3d10_curvo", "C3D10", qc, qe, qn)
# C3D8 distorcido: nos de topo fora do plano (faces nao planas)
nd = dict(nodes)
for n, (x, y, z) in nodes.items():
    if 0 < x < L and abs(z - H) < 1e-12: nd[n] = (x, y, H + 0.3 * ((n % 5) - 2) / 2.0)
    elif 0 < x < L: nd[n] = (x + 0.8 * ((n % 3) - 1), y, z)
write("p_c3d8_distorcido", "C3D8", nd, hexes, nodes)
