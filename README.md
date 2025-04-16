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

1. Külső szuperstruktúra (háromszög) létrehozása, amely tartalmazza az összes bemeneti pontot.
2. Véletlenszerű sorrendbe állítjuk a P pontokat.
3. Minden pontot sorban beillesztünk:
   - Megkeressük azt a háromszöget, amely tartalmazza a pontot.
   - Ha **a pont belül van**:
     - Az adott háromszöget három részre bontjuk.
     - Új éleket hozunk létre a csúcspontból a háromszög csúcsai felé.
     - Legalizáljuk az új éleket
   - Ha **a pont egy élre esik**:
     - Két háromszöget módosítunk.
     - Új éleket húzunk a pontból a háromszögek harmadik csúcspontjához.
     - Legalizáljuk a éleket
4. A végső gráfból eltávolítjuk a szuperstruktúra pontjait és éleit.

---


## 

**Pál Krisztián**  
Újvidéki Egyetem, Természettudományi-Matematikai Kar, hallgató

**Műhelyvezető**:  
Dr. Pintér Róbert

---



---
