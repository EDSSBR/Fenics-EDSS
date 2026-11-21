"""Decks de teste do passo 7 (hiperelastico Neo-Hooke + NLGEOM). Nao edita decks do Luiz/Valdir."""
import os, shutil
R = "/home/valdir/mike-tmp/hyper/c/decks"; os.makedirs(R, exist_ok=True)
SRC = "/home/valdir/mike-tmp/inp-generico/cmp2/decks/beam_hyper.inp"
s = open(SRC).read()
OUTREQ = "*NODE FILE\nU\n*EL PRINT, ELSET=Eall\nS\n*NODE PRINT, NSET=Fixos, TOTALS=YES\nRF\n"
# (a) 0003 beam_hyper com pedidos de S e RF (mesma fisica)
open(f"{R}/beam_hyper.inp", "w").write(s.replace("*NODE FILE\nU\n", OUTREQ))
# (b) sem NLGEOM (o que o ccx faz?)
open(f"{R}/beam_hyper_semnlgeom.inp", "w").write(s.replace("*STATIC, NLGEOM", "*STATIC").replace("*NODE FILE\nU\n", OUTREQ))

KUHN = [(0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6), (0, 5, 1, 6)]
def vol6(a, b, c, d):
    u = [b[i]-a[i] for i in range(3)]; v = [c[i]-a[i] for i in range(3)]; w = [d[i]-a[i] for i in range(3)]
    return u[0]*(v[1]*w[2]-v[2]*w[1]) - u[1]*(v[0]*w[2]-v[2]*w[0]) + u[2]*(v[0]*w[1]-v[1]*w[0])
def grid(nx, ny, nz, L, B, H):
    nid = lambda i, j, k: 1 + i + (nx+1)*(j + (ny+1)*k)
    nodes = {nid(i, j, k): (L*i/nx, B*j/ny, H*k/nz) for k in range(nz+1) for j in range(ny+1) for i in range(nx+1)}
    hexes = [(i, [nid(i, j, k), nid(i+1, j, k), nid(i+1, j+1, k), nid(i, j+1, k),
                  nid(i, j, k+1), nid(i+1, j, k+1), nid(i+1, j+1, k+1), nid(i, j+1, k+1)])
             for k in range(nz) for j in range(ny) for i in range(nx)]
    return nodes, hexes
def tets(nodes, hexes):
    out = []
    for ix, h in hexes:
        for t in KUHN:
            v = [h[q] for q in t]
            if vol6(*[nodes[n] for n in v]) < 0: v[1], v[2] = v[2], v[1]
            out.append((ix, v))
    return out
def quad(nodes, tt):
    nodes = dict(nodes); nxt = max(nodes)+1; mid = {}; out = []
    for ix, v in tt:
        ex = []
        for a, b in ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)):
            k = tuple(sorted((v[a], v[b])))
            if k not in mid:
                mid[k] = nxt; nodes[nxt] = tuple((nodes[v[a]][i]+nodes[v[b]][i])/2 for i in range(3)); nxt += 1
            ex.append(mid[k])
        out.append((ix, v+ex))
    return nodes, out

def write(name, etype, nodes, els, mats, bcs, steps, nset_rf):
    used = {n for _, c in els for n in c}
    with open(f"{R}/{name}.inp", "w") as f:
        f.write("*NODE\n"); [f.write(f"{n}, {x:.10g}, {y:.10g}, {z:.10g}\n") for n, (x, y, z) in sorted(nodes.items()) if n in used]
        f.write(f"*ELEMENT, TYPE={etype}, ELSET=Eall\n")
        for eid, (_, c) in enumerate(els, 1):
            s_ = [str(eid)] + [str(n) for n in c]
            f.write(", ".join(s_[:16]) + ("\n" if len(s_) <= 16 else ",\n" + ", ".join(s_[16:]) + "\n"))
        for m, (card, sel) in mats.items():
            ids = [str(e) for e, (ix, _) in enumerate(els, 1) if sel(ix)]
            f.write(f"*ELSET, ELSET=E_{m}\n"); [f.write(", ".join(ids[i:i+16]) + "\n") for i in range(0, len(ids), 16)]
            f.write(f"*MATERIAL, NAME={m}\n{card}*SOLID SECTION, ELSET=E_{m}, MATERIAL={m}\n")
        allfix = sorted({n for n, _ in bcs})
        f.write("*NSET, NSET=Fixos\n"); [f.write(", ".join(map(str, allfix[i:i+16])) + "\n") for i in range(0, len(allfix), 16)]
        f.write("*BOUNDARY\n"); [f.write(f"{n}, {d}, {d}, 0.0\n") for n, d in bcs]
        for sp, inc, loads in steps:
            f.write(f"*STEP{sp}\n*STATIC\n{inc}\n*CLOAD\n")
            [f.write(f"{n}, {d}, {v}\n") for n, d, v in loads]
            f.write(OUTREQ + "*END STEP\n")

NH = "*HYPERELASTIC, NEO HOOKE\n0.5, 0.04\n"
L = 10.0
nodes, hexes = grid(2, 2, 2, L, L, L)
def sym_bcs(nodes, used):
    b = []
    for n, (x, y, z) in nodes.items():
        if n not in used: continue
        if abs(x) < 1e-12: b.append((n, 1))
        if abs(y) < 1e-12: b.append((n, 2))
        if abs(z) < 1e-12: b.append((n, 3))
    return b
def face_loads(nodes, used, total):
    face = sorted(n for n, (x, y, z) in nodes.items() if abs(x-L) < 1e-12 and n in used)
    return [(n, 1, total/len(face)) for n in face], face
# bloco Neo-Hooke em tracao (simetria em x0/y0/z0), 2 passos: tracao 40 N; depois 25 N + cortante 3 N em y no canto
for etype, (nn, ee) in {"C3D8": (nodes, hexes), "C3D4": (nodes, tets(nodes, hexes))}.items():
    used = {n for _, c in ee for n in c}
    l1, face = face_loads(nn, used, 40.0)
    l2, _ = face_loads(nn, used, 25.0)
    corner = max(face, key=lambda n: (nn[n][1], nn[n][2]))
    l2 = l2 + [(corner, 2, 3.0)]
    write(f"bloco_nh_{etype.lower()}", etype, nn, ee, {"Borracha": (NH, lambda ix: True)}, sym_bcs(nn, used),
          [(", NLGEOM", "0.1, 1.0", l1), (", NLGEOM", "0.1, 1.0", l2)], "Fixos")
# C3D10 tambem (mesma geometria)
qn, qe = quad(nodes, tets(nodes, hexes))
used = {n for _, c in qe for n in c}
l1, face = face_loads(qn, used, 40.0)
write("bloco_nh_c3d10", "C3D10", qn, qe, {"Borracha": (NH, lambda ix: True)}, sym_bcs(qn, used),
      [(", NLGEOM", "0.1, 1.0", l1)], "Fixos")
# viga de aco *ELASTIC com NLGEOM (grande rotacao) -> SVK: 20x1x1 C3D8, 100x5x5, ponta 2000 N em z
nb, hb = grid(20, 1, 1, 100.0, 5.0, 5.0)
used = {n for _, c in hb for n in c}
fix = [(n, d) for n, (x, y, z) in nb.items() if abs(x) < 1e-12 for d in (1, 2, 3)]
tip = sorted(n for n, (x, y, z) in nb.items() if abs(x-100.0) < 1e-12)
AC = {"Aco": ("*ELASTIC\n210000, 0.3\n", lambda ix: True)}
T1 = [(n, 3, 2000.0/len(tip)) for n in tip]
T2 = [(n, 3, 3000.0/len(tip)) for n in tip] + [(tip[0], 2, 150.0)]
write("viga_svk_c3d8", "C3D8", nb, hb, AC, fix, [(", NLGEOM", "0.05, 1.0", T1)], "Fixos")
# semantica do NLGEOM entre passos (o que o ccx faz?)
write("svk_2p_nl_nl", "C3D8", nb, hb, AC, fix, [(", NLGEOM", "0.05, 1.0", T1), (", NLGEOM", "0.05, 1.0", T2)], "Fixos")
write("svk_2p_nl_sem", "C3D8", nb, hb, AC, fix, [(", NLGEOM", "0.05, 1.0", T1), ("", "0.05, 1.0", T2)], "Fixos")
write("svk_2p_sem_nl", "C3D8", nb, hb, AC, fix, [("", "0.05, 1.0", T1), (", NLGEOM", "0.05, 1.0", T2)], "Fixos")
write("svk_2p_nl_no", "C3D8", nb, hb, AC, fix, [(", NLGEOM", "0.05, 1.0", T1), (", NLGEOM=NO", "0.05, 1.0", T2)], "Fixos")
write("viga_svk_c3d4", "C3D4", nb, tets(nb, hb), AC, fix, [(", NLGEOM", "0.05, 1.0", T1)], "Fixos")
print(sorted(os.listdir(R)))
