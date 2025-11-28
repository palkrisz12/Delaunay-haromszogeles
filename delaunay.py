import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import random
import math

# Mivel osztunk a pont ellenorzesnel, esetlegesen nagyon kis eltereseket is figyelembe vesz a szamitogep,
# ezert ezt kicsit onkenyes modon korrigaljuk, letrehozunk egy globalis valtozot, ami a "kozelt" szimulalja
CONST_EPSILON = 1e-12


def compute_super_triangle(points):
    """
    Szuper haromszog letrehozasa
    """
    if not points:
        return [(0, 0), (1, 0), (0, 1)]

    # Bounding box kiszamitasa
    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_y = max(p[1] for p in points)

    # Kozeppont es dimenziok meghatarozasa
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    width = max_x - min_x
    height = max_y - min_y

    # Eset amikor az osszes pont megegyezik
    if width == 0 and height == 0:
        width = height = 1.0

    # Meret faktor
    size = max(width, height) * 10

    point_a = (center_x - 2 * size, center_y - size)
    point_b = (center_x + 2 * size, center_y - size)
    point_c = (center_x, center_y + 2 * size)

    return [point_a, point_b, point_c]

def are_collinear(p1, p2, p3, eps=CONST_EPSILON):
    """
    Parameters
    ----------
    p1,p2,p3
    3 pont, (x,y) formaban
    eps :
        Kozelito ertek, mivel alkalmazunk osztast, a float ertekek felrecsuszhatnak

    Returns
    -------
    TYPE
        A fuggveny celja, hogy megallapitsuk a harom pontukrol, hogy kollinearisak e.
        Az elfajzott haromszogek megakadalyozasanak vegso celjabol
        Kiszamitja a teruletet, ha a terulet kozelit a nullahoz, akkor vagy kollinearis pontjaink vannak,
        vagy elfajzott haromszog

    """
    area = abs((p1[0] * (p2[1] - p3[1]) +
                p2[0] * (p3[1] - p1[1]) +
                p3[0] * (p1[1] - p2[1])) / 2.0)
    return area < eps


def sort_points_ccw(p1, p2, p3):
    """
    3 pontot rendezunk Counter-Clockwise(Oraval ellentetes iranyba), abbol a celbol,
    hogy ne zavarjuk ossze a (insideTriangle es test_incircle) fuggvenyt a pontok egymashoz valo helyzetevel
    """

    def cross(o, a, b):
        # OA × OB Keresztszorzata
        return (a[0] - o[0]) * (b[1] - o[1]) - \
            (a[1] - o[1]) * (b[0] - o[0])

    # Addig probalgatjuk, ameddig nem talaljuk meg a helyes permutaciot
    points = [p1, p2, p3]
    if cross(points[0], points[1], points[2]) > 0:
        return points
    else:
        #
        return [points[0], points[2], points[1]]


class TriangleNode:
    """
        Maga a fa strukturank, alkalmazunk OOP elemeket is. A struktura segitsegevel nyomon kovetjuk a haromszogeinket
        Azert alkalmazzuk ezt a konkret strukturat, hogy felgyorsitsuk a keresest
        Az elek menti illeszkedo haromszogeket el terkep segitsegevel keressuk, egy el ket haromszogre mutat,
        ha hozzaadunk a terkephez egy haromszoget, akkor az elei mind kulcsok lesznek hozza
    """

    edge_map = {}
    root_triangle = None
    triangles = None

    def __init__(self, points=None, root=False):
        if points is None:
            raise ValueError("A nodenak szuksege van ertekekre - inicializalas")

        if len(points) != 3 and root is False:
            raise ValueError("A haromszognek pontosan 3 pontja van. Es nem kunvox burok")

        self.points = points  # Minden TriangleNode level az inicializalaskor
        self.children = None  # Tehat nincsennek children nodejai

    @staticmethod
    def edge_key(p1, p2):
        """Egy el meghatarozasa"""
        return tuple(sorted((p1, p2), key=lambda p: (p[0], p[1])))

    def register_triangle(self):
        """Hozzaadjuk a haromszoget az el terkepbe"""
        a, b, c = self.points
        for edge in [(a, b), (b, c), (c, a)]:
            key = TriangleNode.edge_key(*edge)
            TriangleNode.edge_map.setdefault(key, set()).add(self)

    def unregister_triangle(self):
        """Eltavolitjuk a haromszoget az el terkepbol"""
        a, b, c = self.points
        for edge in [(a, b), (b, c), (c, a)]:
            key = TriangleNode.edge_key(*edge)
            if key in TriangleNode.edge_map:
                triangles = TriangleNode.edge_map[key]
                triangles.discard(self)
                if not triangles:  # Ures nodeok eltavolitasa
                    del TriangleNode.edge_map[key]

    def add_children(self, new_children):
        """A metodus megvaltoztatja egy node children ertekeit, vagyis rendel hozzajuk,
           A levelbol belso nodeot alakit a levelbol
        """

        if not self.is_leaf():
            raise ValueError("Csak leaf node-ok kaphatnak gyerekeket")

        if any(not isinstance(c, TriangleNode) for c in new_children):
            raise TypeError("Csak TriangleNodeok lehet gyerekek")

        # Megvizsgaljuk a children haromszogeket, hogy elfajzottak e, csak a megfelelo haromszogek lehetnek
        legit_children = []
        for child in new_children:
            if not are_collinear(child.points[0], child.points[1], child.points[2]):
                TriangleNode.register_triangle(child)
                legit_children.append(child)

        self.children = legit_children

        # Eltavolitjuk a levelek halmazabol a haromszogunket
        if self is not TriangleNode.root_triangle:
            TriangleNode.unregister_triangle(self)
        # Hozzaadjuk a levelek halmazahoz az uj haromszogeket

    def is_leaf(self):
        """Megvizsgaljuk a TriangleNodeot, hogy level e"""
        return self.children is None

    def plot_leaves(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()

        default_kwargs = {
            'edgecolor': 'black',
            'linewidth': 1.1,
            'alpha': 0.7
        }
        default_kwargs.update(kwargs)

        a, b, c = TriangleNode.root.points
        super_vertices = {tuple(a), tuple(b), tuple(c)}

        patches = []

        all_triangles = set()
        for triangles in TriangleNode.edge_map.values():
            all_triangles.update(triangles)
        for leaf in all_triangles:
            if not any(tuple(p) in super_vertices for p in leaf.points):
                patches.append(Polygon(leaf.points))

        default_kwargs = {
            'facecolor': 'skyblue',
            'edgecolor': 'navy',
            'linewidth': 1.5,
            'alpha': 0.7
        }
        default_kwargs.update(kwargs)

        collection = PatchCollection(patches, **default_kwargs)
        ax.add_collection(collection)
        ax.autoscale_view()
        ax.set_aspect('equal')
        return ax

    def __str__(self):
        """A TriangleNode string reprezentalasa, debuggolas miatt"""
        formatted_points = [
            f"({x:.2f}, {y:.2f})" for x, y in self.points
        ]
        if self.is_leaf():
            return f"LeafNode: [{', '.join(formatted_points)}]"
        else:
            child_count = len(self.children)
            return f"InternalNode: {child_count} children; [{', '.join(formatted_points)}]"

    def __repr__(self):
        """Kiiratas"""
        return self.__str__()

    def insideTriangle(self, point):
        """
        A baricentrikus koordinatak segitsegevel megvizsgaljuk a haromszog es a pont viszonyat
        egymashoz kepest, kezenfekvo alkalmaznunk, mivel a segitsegukkel pontosan meg tudjuk allapitani,
        hogy hol van a pont a haromszoghoz kepest, benne, rajta kivul vagy pedig konkretan melyik elen
        P = A + s(B-A) + t(C-A) kepletet

        """

        if point is not None:
            p0, p1, p2 = self.points

            a, b, c = sort_points_ccw(p0, p1, p2)
            p0, p1, p2 = a, b, c

            area = 0.5 * (-p1[1] * p2[0] + p0[1] * (-p1[0] + p2[0]) + p0[0] * (p1[1] - p2[1]) + p1[0] * p2[1])

            # print(area)
            if abs(area) < CONST_EPSILON:
                raise ValueError("Harom kollinearis pontot vizsgalunk, az eredmeny helytelen")

            s = 1 / (2 * area) * (
                        p0[1] * p2[0] - p0[0] * p2[1] + (p2[1] - p0[1]) * point[0] + (p0[0] - p2[0]) * point[1])
            t = 1 / (2 * area) * (
                        p0[0] * p1[1] - p0[1] * p1[0] + (p0[1] - p1[1]) * point[0] + (p1[0] - p0[0]) * point[1])

            if t > 0 and s > 0 and s + t > 1 - CONST_EPSILON and s + t < 1 + CONST_EPSILON:

                # a pont p1 es p2 pontokat osszekoto elen helyezkedik el, visszaadjuk az elet
                # a fuggveny eredmenyet ismerve dontjuk el a kovetkezo lepesunket
                return [p1, p2]
            elif s > 0 and t > 0 and 1 - s - t > 0:

                # a pont benne van a haromszogben
                return True
            elif s > CONST_EPSILON and s < 1 - CONST_EPSILON and abs(t) < CONST_EPSILON:

                # a pont p0 es p1 pontok elein helyezkedik el
                return [p0, p1]
            elif t > CONST_EPSILON and t < 1 - CONST_EPSILON and abs(s) < CONST_EPSILON:

                # a pont p0 es p2 pontok elein helyezkedik el
                return [p0, p2]
            else:
                # a pont nincs a haromszogben
                return False

    def test_incircle(self, p):
        if p is not None:
            """
            Ellenorizzuk, hogy egy negyedik pont benne van e a haromszogunk korulirhato koreben,
            kesobb szuksegunk lesz ra amikor az eleket teszteljuk, egy determinanst kiszamolunk, majd megvizsgaljuk az ertek elojelet
            """
            a, b, c = self.points
            triangle = sort_points_ccw(a, b, c)
            a, b, c = triangle
            ax = a[0] - p[0]
            ay = a[1] - p[1]
            bx = b[0] - p[0]
            by = b[1] - p[1]
            cx = c[0] - p[0]
            cy = c[1] - p[1]

            return ((ax * ax + ay * ay) * (bx * cy - cx * by) - (bx * bx + by * by) * (ax * cy - cx * ay) + (
                        cx * cx + cy * cy) * (ax * by - bx * ay)) > CONST_EPSILON

    def find_adjacent_triangle(self, edge):
        """Illeszkedo haromszogek keresese, csupan a terkepben magaban keresunk, nincs szukseg O(n) keresesre"""
        key = TriangleNode.edge_key(*edge)
        triangles = TriangleNode.edge_map.get(key, set())
        for tri in triangles:
            if tri is not self:
                return tri
        return None

    def search_third_point(self, edge):
        """
        Seged fuggveny, megkeresi a haromszoget alkoto harmadik pontot, vagyis azt amelyik nem alkotja a megfigyelt elet
        """
        for p in self.points:
            if p not in edge:
                return p
        return None

    def legalize_edge(self, edge):
        """
        A fuggveny megkeresi az illeszkedo haromszoget az el menten,
        Elofordulhat, hogy a superharomszog egy kulso, illeszkedo haromszoget keressuk
        (Vagyis a Szuperharomszogunk egyik oldala menten)
        Ilyenkor nem letezik kulso haromszog, ezt figyelembe kell vennunk

        Ha letezik ilyen belso haromszog, vagyis a kereses nem None erteket ad vissza, akkor megkeressuk
        a harmadik pontjat(OT_p), amely nem alkotja az elet, meghivjuk a Delaunay-feltetel menten
        mukodo fuggvenyunket, amely megvizsgalja, hogy az OT_p bele esik e a self haromszogunk korulirhato korebe,
        ha igen, akkor tudjuk, hogy az edge, amelyre meghivtuk illegalis es meg kell forditanunk,
        amennyiben megforditottunk, meg kell vizsgalnunk ket tovabbi elet, hogy illegalisak lesznek
        Az algoritmus elonye, hogy legtobbszor csak a lokalis clusterben talalhato eleket kell atvizsgalni
        """
        OtherTriangle = self.find_adjacent_triangle(edge)
        # print(edge)
        # print(OtherTriangle)
        if OtherTriangle is not None:
            # Az illeszkedo haromszog harmadik pontja - OtherTriangle_point
            OT_p = OtherTriangle.search_third_point(edge)
            # Megfigyelt haromszog harmadik pontja - CurrentTriangle_point
            CT_p = self.search_third_point(edge)
            if self.test_incircle(OT_p):
                # print("True " + str(OT_p) + " "+str(self))
                triangle1 = TriangleNode([edge[0], OT_p, CT_p])
                triangle2 = TriangleNode([edge[1], OT_p, CT_p])
                new_children = [triangle1, triangle2]
                self.add_children(new_children)
                OtherTriangle.add_children(new_children)
                triangle1.legalize_edge([edge[0], OT_p])
                triangle2.legalize_edge([edge[1], OT_p])

    # a fuggveny hozzaadja az adott pontot a haromszogeleshez
    def add_point_inside(self, p):
        """
        A fuggveny a pont beszuras egyik formaja, ha tudjuk, hogy a pontnunk a haromszog belsejebe esik,
        akkor harom uj haromszoget hozunk letre a segitsegevel, vagyis 3 szeletre osztjuk a nagy haromszogunket
        """
        # Amennyiben benne vagyunk a haromszogben letrehozunk 3 uj haromszoget, majd megvizsgaljuk, hogy az eddig letezo el legalis-e
        triangle1 = TriangleNode([self.points[0], self.points[1], p])
        triangle2 = TriangleNode([self.points[1], self.points[2], p])
        triangle3 = TriangleNode([self.points[2], self.points[0], p])

        new_children = [triangle1, triangle2, triangle3]

        self.add_children(new_children)
        triangle1.legalize_edge([self.points[0], self.points[1]])
        triangle2.legalize_edge([self.points[1], self.points[2]])
        triangle3.legalize_edge([self.points[2], self.points[0]])

    def add_point_edge(self, p, edge):
        """
        A pont beszuras masik formaja, ha a fuggvenyunk az elre esik, akkor megkeressuk azt a haromszoget,
        amelyik osztja az elet, es ket uj haromszogra bontjuk mindket vizsgalt haromszogunket, majd megvizsgaljuk,
        hogy az ujonnan beszurt elek legalisak-e
        """
        a, b = edge
        c = self.search_third_point(edge)
        triangle1 = TriangleNode([a, p, c])
        triangle2 = TriangleNode([b, p, c])

        new_children = [triangle1, triangle2]

        self.add_children(new_children)

        triangle1.legalize_edge([a, c])
        triangle2.legalize_edge([b, c])

    def search_and_add(self, p):
        """
        Kette bontottam a search_and_add fuggvenyt
        A javitas lenyege, hogy csak egyszer szamoljuk ki a valaszt, ne tobbszor

        """
        answer = self.insideTriangle(p)
        self.search_and_add_helper(p, answer)

    # A fuggveny megkeresi az adott haromszoget, majd ha a felteteleknek megfelel hozzadja
    def search_and_add_helper(self, p, answer):
        """
        Javitott kereses a faban,
        Maga a valasz is resze a fuggvenynek, amit a for ciklusban kerdezunk meg,
        igy csupan egyszer tesszuk fel
        Ha benne van, megkerdezzuk, hogy level-e,
        Ha Level akkor beszurjuk a pontot, vagy a haromszog belsejebe, vagy egy el menten felbontjuk a haromszoget
        Ha nem level akkor atnezzuk a gyerekeket, egyesevel megkerdezzuk,
        Ha valamelyik benne van, meghivjuk ra a fuggvenyunket, amely ugyanezt elvegzi ra
        Igy nem kell az egesz fat bejarnunk, hanem eleg csupan az agak menten vegigsetalni

        """
        if answer is not False:
            # p benne van a faban
            if self.is_leaf():
                # vagy egyik, vagy masik lefut
                if answer is True:
                    self.add_point_inside(p)
                    return
                else:
                    # a valasz egy el
                    # print("elmenti kereses")
                    adj = self.find_adjacent_triangle(answer)
                    if adj:
                        self.add_point_edge(p, answer)
                        adj.add_point_edge(p, answer)
                        return
            else:
                # internal node: a gyerekek ertekeit megvizsgaljuk
                if self.children is not None:
                     for child in self.children:
                        answer_new = child.insideTriangle(p)
                        if answer_new is not False:
                            child.search_and_add_helper(p, answer_new)
                            return


import time
import psutil
class DelaunayTriangulation:
    def __init__(self, points):
        """Inicializaljuk a haromszoget es a pontjait"""
        self.points = points
        self.root_triangle = None
        self.build()

    def build(self):
        """Letrehozzuk magat a haromszogelest"""

        self.root_triangle = compute_super_triangle(self.points)
        TriangleNode.root = TriangleNode(self.root_triangle)
        TriangleNode.edge_map.clear()
        TriangleNode.register_triangle(TriangleNode.root)
        for p in self.points:
            TriangleNode.root.search_and_add(p)


    def get_leaf_triangles(self):
        """Visszaadjuk a haromszogelesben talalhato haromszogeket, amelyek a muveletek vegen megtalalhatoak"""
        all_triangles = set()
        for tris in TriangleNode.edge_map.values():
            for tri in tris:
                if not tri.children:
                    all_triangles.add(tri)
        return all_triangles

    def cts(self):
        """
        A fuggveny megszamolja a haromszogeket, keplet segitsegevel

        edge_map: dict
            Meghatarozzuk a hany el talalhato az el terkepben, amelyek nem incidensek a szuperstrukturaval

        Returns:
            int: A haromszogek szama
        """
        def count_edges(edge_map):

            a, b, c = TriangleNode.root.points
            super_vertices = {tuple(a), tuple(b), tuple(c)}
            count = 0
            for (u, v) in edge_map.keys():
                if not any(tuple(p) in super_vertices for p in [u,v]):
                    count += 1
            return count

        edge_map = TriangleNode.edge_map
        total_edges = count_edges(edge_map)
        triangles = (2 * total_edges - 3) // 3
        return triangles


    def plotTrian(self):
        plt.figure(figsize=(10, 10))
        TriangleNode.root.plot_leaves()
        plt.title("Delaunay-trianguláció")
        plt.xlabel(str(len(points)) + " pontra")
        plt.show()
        plt.close()

    def destroy(self):
        TriangleNode.root_triangle = None
        TriangleNode.edge_map.clear()


def gen_uniform(n, rng, low=0.0, high=1.0):
    return [(rng.uniform(low, high), rng.uniform(low, high)) for _ in range(n)]


def gen_grid(n, side=None):
    # make ~square grid with <= n points
    if side is None:
        side = int(math.sqrt(n))
    xs, ys = np.linspace(0.0, 1.0, side), np.linspace(0.0, 1.0, side)
    pts = [(float(x), float(y)) for x in xs for y in ys]
    return pts[:n]

import numpy as np

import os
import sys
import time
import math
import csv
import random
import importlib.util
from contextlib import contextmanager


def gen_spiral(n, rng, a=0.02, b=0.15, turns=3.5):
    # Archimedean spiral with small jitter
    thetas = np.linspace(0, 2*math.pi*turns, n)
    pts = []
    for th in thetas:
        r = a + b*th
        x = r*math.cos(th) + rng.normalvariate(0, 0.002)
        y = r*math.sin(th) + rng.normalvariate(0, 0.002)
        pts.append((x, y))
    # normalize to [0,1]^2 for fairness
    xs, ys = zip(*pts)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    pts = [((x - xmin)/(xmax - xmin + 1e-12), (y - ymin)/(ymax - ymin + 1e-12)) for (x, y) in pts]
    return pts

def gen_clusters(n, rng, k=4, spread=0.05):
    centers = [(rng.uniform(0.1, 0.9), rng.uniform(0.1, 0.9)) for _ in range(k)]
    out = []
    per = max(1, n // k)
    for (cx, cy) in centers:
       for _ in range(per):
           out.append((rng.normalvariate(cx, spread), rng.normalvariate(cy, spread)))
    return out[:n]

if __name__ == "__main__":
    # points = [(1,2),(3,8), (5,4), (7,2), (1,4), (3,6), (5,2)]
    # points = [(1,1), (2,2), (3,3), (4,4), (5,5),(5,1), (6,2), (7,3), (4,0), (2,0), (0,2) ,(0,1)]

    seed = random.seed(8)

    points = gen_uniform(500, random.Random(seed))
    t0 = time.perf_counter()
    dt = DelaunayTriangulation(points)
    print("Plot time: " + str(time.perf_counter() - t0))
    dt.plotTrian()
    print(dt.cts())
    dt.destroy()







