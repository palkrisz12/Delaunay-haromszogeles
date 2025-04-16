import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import random

#Mivel osztunk a pont ellenorzesnel, esetlegesen nagyon kis eltereseket is figyelembe vesz a szamitogep, 
#ezert ezt kicsit onkenyes modon korrigaljuk, letrehozunk egy globalis valtozot, ami a "kozelt" szimulalja
CONST_EPSILON = 1e-8

def compute_super_triangle(points):
    """
    A fuggveny letrehoz egy Szuper haromszoget, amely eleg nagy ahhoz, hogy ne zavarjon be a belso strukturak haromszogelesebe,
    valamint tartalmazza az osszes pontunkat, ez lesz a root haromszogunk, amelyet fel fogunk kesobb darabolni

    """
    min_x = min(points, key=lambda p: p[0])[0]
    max_x = max(points, key=lambda p: p[0])[0]
    min_y = min(points, key=lambda p: p[1])[1]
    max_y = max(points, key=lambda p: p[1])[1]
           
    x_distance = max_x - min_x
    y_distance = max_y - min_y
    
    point_a = (min_x + 2.5 * x_distance, min_y + 0.5 * y_distance)  
    point_b = (min_x + 0.5 * x_distance, min_y + 2.5 * y_distance)  
    point_c = (min_x - 2.0 * x_distance, min_y - 2.0 * y_distance)  
            
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
    area = abs((p1[0]*(p2[1] - p3[1]) +
                p2[0]*(p3[1] - p1[1]) +
                p3[0]*(p1[1] - p2[1])) / 2.0)
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
    """
    def __init__(self, points=None):
        if points is None:
            raise ValueError("A nodenak szuksege van ertekekre - inicializalas")

        if len(points) != 3:
            raise ValueError("A haromszognek pontosan 3 pontja van.")
            
       
        self.points = points  # Minden TriangleNode level az inicializalaskor
        self.children = None  # Tehat nincsennek children nodejai
        self.parent = None  # Inicializalaskor parent nodeot sem adunk meg, mivel szuksegunk lesz a root node megkulonboztetesere
       


    def add_children(self, new_children):
        """A metodus megvaltoztatja egy node children ertekeit, vagyis rendel hozzajuk,
           A levelbol belso nodeot alakit a levelbol
        """
        if not self.is_leaf():
            raise ValueError("Csak leaf node-ok kaphatnak gyerekeket")
            
        if any(not isinstance(c, TriangleNode) for c in new_children):
            raise TypeError("Csak TriangleNodeok lehet gyerekek")
            
        #Megvizsgaljuk a children haromszogeket, hogy elfajzottak e, csak a megfelelo haromszogek lehetnek 
        legit_children = []
        for child in new_children:
            if not are_collinear(child.points[0], child.points[1], child.points[2]):
                legit_children.append(child)
        
        self.children = legit_children
        
        #Az osszes megfelelo gyerekhez rendelunk szulot
        for child in legit_children:   
            child.parent = self

    def is_leaf(self):
        """Megvizsgaljuk a TriangleNodeot, hogy level e"""
        return self.children is None 

    
    def get_all_leaves(self):
        """A fuggveny visszaadja a fa egyik konkret aganak osszes levelet"""
        if self.is_leaf():
            return [self]
        return [leaf for child in self.children for leaf in child.get_all_leaves()]

    def plot_leaves(self, ax=None, **kwargs):
        """PatchCollection alkalmazasaval megrajzoljuk az osszes haromszoget, amire szuksegunk van"""
        if ax is None:
            ax = plt.gca()
        
        #Default stilus
        default_kwargs = {
            'facecolor': 'skyblue',
            'edgecolor': 'navy',  
            'linewidth': 1.5,
            'alpha': 0.7
        }
        default_kwargs.update(kwargs)
        
        
        def any_points_equal(a, b, c, p1, p2, p3):
            """
            Halmaz alkalmazasaval megvizsgaljuk, hogy a 6 pont kozul barmelyik ugyanaz-e
            Az a, b, c egy konkret haromszog csucsai, meg a p1, p2, p3 egy masik haromszoge
            Az eredeti Szuperharomszog miatt

            """
            points = [a, b, c, p1, p2, p3]
            return len(points) != len(set(points))
               
        def get_all_needed_leaves_helper(TriangleNode, a, b, c):
            """
            Rekurzioval megkeressuk az osszes olyan levelet, amely megfelel a feltetelnek,
            jelen esetben a levelek nem tartalmazhatnak pontot a szuperharomszogbol
            """
            if TriangleNode.is_leaf(): 
                if any_points_equal(a, b, c, TriangleNode.points[0], TriangleNode.points[1], TriangleNode.points[2]) == False:
                    return [TriangleNode]
                return []
            else:
                leaves = []
                for child in TriangleNode.children:
                    leaves.extend(get_all_needed_leaves_helper(child, a, b ,c))
                return leaves
        
        def get_all_needed_leaves(root):
            """
            A fo fuggveny, amely gyakorlatilag fejleckent funkcional
            """
            a,b,c = root.points
            return get_all_needed_leaves_helper(root, a,b,c)
        
        root = self._get_root()
        leaves = get_all_needed_leaves(root)

        patches = [Polygon(leaf.points) for leaf in leaves]
        
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
            # Format points with 2 decimal places for readability
            
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
               
               a, b, c = sort_points_ccw(p0,p1,p2)
               p0,p1,p2 = a, b, c
               
               area = 0.5 * (-p1[1]*p2[0] + p0[1]*(-p1[0] + p2[0]) + p0[0]*(p1[1] - p2[1]) + p1[0]*p2[1])
               
                
               if abs(area) < CONST_EPSILON:
                   raise ValueError("Harom kollinearis pontot vizsgalunk, az eredmeny helytelen")
                    
               s = 1/(2*area) * (p0[1]*p2[0] - p0[0]*p2[1] + (p2[1] - p0[1])*point[0] + (p0[0] - p2[0]) * point[1])
               t = 1/(2*area) * (p0[0]*p1[1] - p0[1]*p1[0] + (p0[1] - p1[1])*point[0] + (p1[0] - p0[0]) * point[1])
               #print(s, t)
               if t > 0 and s > 0 and s+t > 1-CONST_EPSILON and s+t < 1+CONST_EPSILON:
                   #a pont p1 es p2 pontokat osszekoto elen helyezkedik el, visszaadjuk az elet
                   #a fuggveny eredmenyet ismerve dontjuk el a kovetkezo lepesunket
                   return [p1, p2]
               elif s > 0 and t > 0 and 1-s-t > 0:
                   #a pont benne van a haromszogben
                   return True
               elif s > 0 and s < 1 and abs(t) < CONST_EPSILON:
                   #a pont p0 es p1 pontok elein helyezkedik el
                   return [p0, p1]
               elif t > 0 and t < 1 and abs(s) < CONST_EPSILON:
                   #a pont p0 es p2 pontok elein helyezkedik el
                   return [p0, p2]
               else:
                   #a pont nincs a haromszogben
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
            ax = a[0]-p[0]
            ay = a[1]-p[1]
            bx = b[0]-p[0]
            by = b[1]-p[1]
            cx = c[0]-p[0]
            cy = c[1]-p[1]
            
            return ((ax*ax + ay* ay) * (bx*cy - cx*by) - (bx*bx + by*by) * (ax*cy -cx*ay) + (cx*cx + cy*cy) * (ax*by-bx*ay)) > 0
   
    def find_adjacent_triangle(self, edge):
       """Megkeressuk egy konkret haromszog illeszkedo haromszoget"""
       root = self._get_root()
       leaves = root.get_all_leaves()
       for l in leaves:
           if l is not self:
               if edge[0] in l.points and edge[1] in l.points:
                   return l
       return None
     

     
    def _get_root(self):
        """Amikor a leveleket keressuk, mindig szuksegunk van a rootra, a fuggveny celja, 
        hogy felmasszon a rootig"""
        current = self
        while current.parent is not None:
            current = current.parent
        return current
   
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
        #print(edge)
        #print(OtherTriangle)
        if OtherTriangle is not None:
            #Az illeszkedo haromszog harmadik pontja - OtherTriangle_point
            OT_p = OtherTriangle.search_third_point(edge)
            #Megfigyelt haromszog harmadik pontja - CurrentTriangle_point
            CT_p = self.search_third_point(edge)
            if self.test_incircle(OT_p):
                triangle1 = TriangleNode([edge[0], OT_p, CT_p])
                triangle2 = TriangleNode([edge[1], OT_p, CT_p])
                new_children = [triangle1, triangle2]
                self.add_children(new_children)
                OtherTriangle.add_children(new_children)
                triangle1.legalize_edge([edge[0], OT_p])
                triangle2.legalize_edge([edge[1], OT_p])
                  
    
    #a fuggveny hozzaadja az adott pontot a haromszogeleshez
    def add_point_inside(self, p):
        """
        A fuggveny a pont beszuras egyik formaja, ha tudjuk, hogy a pontnunk a haromszog belsejebe esik,
        akkor harom uj haromszoget hozunk letre a segitsegevel, vagyis 3 szeletre osztjuk a nagy haromszogunket
        """
        #Amennyiben benne vagyunk a haromszogben letrehozunk 3 uj haromszoget, majd megvizsgaljuk, hogy az eddig letezo el legalis-e
        triangle1 = TriangleNode([self.points[0], self.points[1], p])
        triangle2 = TriangleNode([self.points[1], self.points[2], p])
        triangle3 = TriangleNode([self.points[0], self.points[2], p])
        
        new_children = [triangle1, triangle2, triangle3]
        
        self.add_children(new_children)
        
        triangle1.legalize_edge([self.points[0], self.points[1]])
        triangle2.legalize_edge([self.points[1], self.points[2]])
        triangle3.legalize_edge([self.points[0], self.points[2]])
        
    
    
    
    def add_point_edge(self, p, edge):
        """
        A pont beszuras masik formaja, ha a fuggvenyunk az elre esik, akkor megkeressuk azt a haromszoget, 
        amelyik osztja az elet, es ket uj haromszogra bontjuk mindket vizsgalt haromszogunket, majd megvizsgaljuk,
        hogy az ujonnan beszurt elek legalisak-e
        """
        a,b = edge
        c = self.search_third_point(edge)
        triangle1 = TriangleNode([a, p, c])
        triangle2 = TriangleNode([b, p, c])
            
        new_children = [triangle1, triangle2]
        
        self.add_children(new_children)
        
        triangle1.legalize_edge([a, c])
        triangle2.legalize_edge([b, c])
    
    
    
    #A fuggveny megkeresi az adott haromszoget, majd ha a felteteleknek megfelel hozzadja
    def search_and_add(self, p):
        """
        A fo fuggvenyunk, ha a vizsgalt haromszogunk level, akkor lekerdezzuk, hogy a pontunk benne van-e,
        ha nincs, akkor megvizsgaljuk a gyerekeit, gyakorlatilag bejarjuk a fa strukturankat,
        ha benne van, akkor beszurjuk
        ha a valasz nem hamis, akkor tudjuk, hogy egy elen vagyunk, azert meghivjuk a fuggvenyt,
        amely beszurja a pontot az ere
        """
        if self.is_leaf():
            answer = self.insideTriangle(p)
            if answer == True:
                self.add_point_inside(p)
            elif answer!= False and len(answer) == 2:
                adjTriangle = self.find_adjacent_triangle(answer)
                if adjTriangle is not None:
                    #Ha letezik, akkor az uj haromszogunkhoz is hozzaadjuk a pontot, levelekre bontjuk
                    self.add_point_edge(p, answer)
                    adjTriangle.add_point_edge(p, answer)
        else:
            for c in self.children:
                c.search_and_add(p)
    




def generate_unique_integer_points(n, x_range=(0, 100), y_range=(0, 100)):
    """
    Generalunk n darab egyedi integer koordinataju pontot egy megadott intervallumban, a pontok nincsennek garantalva, hogy altalanos helyzetben lesznek,
    viszont tobbe-kevesbe megfeleloen haromszogelhetoek

    Parameterek:
    - x_range: (xmin, xmax)
    - y_range: (ymin, ymax) 

    """
    xmin, xmax = x_range
    ymin, ymax = y_range

    all_points = [(x, y) for x in range(xmin, xmax + 1) 
                          for y in range(ymin, ymax + 1)]

    if n > len(all_points):
        raise ValueError(f"Nem lehetseges {n} egyedi pont generalasa a megadott intervallumban")

    return random.sample(all_points, n)

"""
A Delaunay-haromszogeles feltetelt ad egy altalanos pozicioban elhelyezkedo ponthalmaz haromszogelesehez,
Akkor talalhato egy ponthalmaz altalanos pozicioban

i. ha barmely megfigyelt negy pont sem talalhato ugyanazon a korvonalon,

Ha teljesul a feltetel, akkor altalanos pozicioban van a haromszogelesunk, es megfelelo Delaunay-triangulaciot generalhatunk ra,
a kovetkezo fuggveny egy ilyen altalanosan pozicioban levo ponthalmazt igyekszik generalni

"""


def are_cocircular(p1, p2, p3, p4):
    """
    Megvizsgaljuk, hogy negy pont ugyanarra a korvonalra esik-e
    """
    def det(a, b, c, d):
        return (
            a[0]*(b[1]*c[2] + b[2]*d[1] + c[1]*d[2] - c[2]*d[1] - b[1]*d[2] - c[1]*b[2])
            - a[1]*(b[0]*c[2] + b[2]*d[0] + c[0]*d[2] - c[2]*d[0] - b[0]*d[2] - c[0]*b[2])
            + a[2]*(b[0]*c[1] + b[1]*d[0] + c[0]*d[1] - c[1]*d[0] - b[0]*d[1] - c[0]*b[1])
        )

    def to_homogeneous(p):
        x, y = p
        return (x, y, x**2 + y**2)

    m = [to_homogeneous(p) for p in [p1, p2, p3, p4]]
    return det(m[0], m[1], m[2], m[3]) == 0

import itertools

def is_in_general_position(points):
    """
    Megvizsgaljuk, hogy a ponthalmazunk altalanos pozicioban van-e, vagyis nincs negy kocirkularis, elegge szamitas igenyes
    feladat, viszont garantalt a helyes megoldas
    """
    # Kocirkularis negyesek kivizsgalasa
    for quadruplet in itertools.combinations(points, 4):
        if are_cocircular(*quadruplet):
            return False

    return True

def generate_unique_random_points(n, x_range=(0, 100), y_range=(0, 100)):
    """
    A fuggveny pontokat general, amelyek csak akkor kerulnek bele a return listbe- ha altalanos pozicioban vannak,
    meg inkabb szamitasigenyesebb feladat


    Parameterek:
        - x_range: (xmin, xmax)
        - y_range: (ymin, ymax) 


    """
    points = set()  # Halmazokkal dolgozunk, mivel a halmazok csak egyedi elemeket tartalmazhatnak
    while len(points) < n:
        new_point = (random.randint(x_range[0], x_range[1]), random.randint(y_range[0], y_range[1]))
        
        temp_points = list(points) + [new_point]
        
        if is_in_general_position(temp_points):
            points.add(new_point)  # Ha a pont megfelel a feltetelnek, hozzaadjuk a halmazunkhoz

    return list(points)  

if __name__ == "__main__":
    
    points = [(9, 10), (-1, -5), (2, 1), (-10, 10), (2, -3), (-7, 1), (4, 6), (9, -4), (1, -3), (-5, 5), (5, -7), (-3, 5)]
    triangle = compute_super_triangle(points)
    
    root = TriangleNode(triangle)
    
    random.shuffle(points)
    
    for p in points:
        root.search_and_add(p)
    
    plt.figure(figsize=(8, 6))
    root.plot_leaves()
    plt.title("Delaunay-trianguláció")
    plt.show()
            
        
    
    
    
    
    
    
    
    
    