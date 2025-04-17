![ek-logo-szoveggel](https://github.com/user-attachments/assets/b69ccf55-586c-4fc7-a8fc-c51f338b2076)



# Delaunay-háromszögelés Pythonban

Ez a projekt az **Európa Kollégium Informatika-műhelyének** keretén belül készült.

A projekt célja egy **Python nyelvű implementáció** készítése, amely **Delaunay-háromszögelést** végez a **Randomized Incremental Construction (RIC)** algoritmus segítségével.

##  Az algoritmus rövid ismertetése

**Input**:  
P = (x, y) koordinátapárokból álló pont halmaz  
**Output**:  
A P halmazra készített Delaunay-háromszögelés

### Lépések:

INPUT. P ponthalmaz, amely n darab (x,y) koordináta párokból áll.
OUTPUT. Egy matplotlib grafikon, amely megrajzolja az algoritmus futása során létrehozott háromszögeket
	Létrehozzuk a külső szuperstruktúrát, amely elég nagy ahhoz, hogy ne zavarja a pontok közötti háromszögelés legalitását
	Véletlenszerű sorrendbe állítjuk a P ponthalmazt
	Létrehozunk egy fa struktúrát (Directed Acyclic Graph), amelynek a gyökere lesz a szuperstruktúránk
	A P ponthalmaz elemeit egyesével beszúrjuk a háromszögelésünkbe
	Megvizsgáljuk, hogy a pontunk benne van-e a háromszögünkben a baricentrikus koordináták segítségével
	Igen: Megvizsgáljuk, hogy a háromszögünk levél-e
Igen: Megvizsgáljuk, hogy a háromszögünk tartalmazza e a belsejében a pontot 
Igen: Beszúrjuk a pontot, három levélértéket kötünk a háromszöghöz, legalizáljuk az eredeti háromszög élet.
Nem: A pontunk egy élen helyezkedik el, megkeressük az illeszkedő háromszöget, felosztjuk, létrehozzuk a leveleket, majd legalizáljuk az eredeti háromszögek éleit
Nem:	Megvizsgáljuk a háromszögeink children listáját, rekurzíven újra meghívjuk a függvényt arra a children értékre, amelyben megtalálható a pont
	Kiíratáskor végigfut a függvény a fán és megrajzolja az összes háromszöget. Vége.


---


## 

**Pál Krisztián**  
Újvidéki Egyetem, Természettudományi-Matematikai Kar, hallgató

**Műhelyvezető**:  
Dr. Pintér Róbert

---



---
