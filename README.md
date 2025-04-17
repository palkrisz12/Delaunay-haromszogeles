![ek-logo-szoveggel](https://github.com/user-attachments/assets/b69ccf55-586c-4fc7-a8fc-c51f338b2076)



# Delaunay-háromszögelés Pythonban

Ez a projekt az **Európa Kollégium Informatika-műhelyének** keretén belül készült.

A projekt célja egy **Python nyelvű implementáció** készítése, amely **Delaunay-háromszögelést** végez a **Randomized Incremental Construction (RIC)** algoritmus segítségével.

##  Az algoritmus rövid ismertetése

<html lang="hu">
<head>
  <meta charset="UTF-8">
  <title>Delaunay algoritmus leírás</title>
  
</head>
<body>

<h2>INPUT</h2>
<p>P ponthalmaz, amely n darab (x,y) koordináta párokból áll.</p>

<h2>OUTPUT</h2>
<p>Egy matplotlib grafikon, amely megrajzolja az algoritmus futása során létrehozott háromszögeket</p>

<ol>
  <li>Létrehozzuk a külső szuperstruktúrát, amely elég nagy ahhoz, hogy ne zavarja a pontok közötti háromszögelés legalitását</li>
  <li>Véletlenszerű sorrendbe állítjuk a P ponthalmazt</li>
  <li>Létrehozunk egy fa struktúrát (Directed Acyclic Graph), amelynek a gyökere lesz a szuperstruktúránk</li>
  <li>
    A P ponthalmaz elemeit egyesével beszúrjuk a háromszögelésünkbe
    <ul>
      <li>Megvizsgáljuk, hogy a pontunk benne van-e a háromszögünkben a baricentrikus koordináták segítségével</li>
      <li>Igen: Megvizsgáljuk, hogy a háromszögünk levél-e
        <ul>
          <li>Igen: Megvizsgáljuk, hogy a háromszögünk tartalmazza e a belsejében a pontot 
            <ul>
              <li>Igen: Beszúrjuk a pontot, három levélértéket kötünk a háromszöghöz, legalizáljuk az eredeti háromszög élet.</li>
              <li>Nem: A pontunk egy élen helyezkedik el, megkeressük az illeszkedő háromszöget, felosztjuk, létrehozzuk a leveleket, majd legalizáljuk az eredeti háromszögek éleit</li>
            </ul>
          </li>
          <li>Nem: Megvizsgáljuk a háromszögeink children listáját, rekurzíven újra meghívjuk a függvényt arra a children értékre, amelyben megtalálható a pont</li>
        </ul>
      </li>
    </ul>
  </li>
  <li>Kiíratáskor végigfut a függvény a fán és megrajzolja az összes háromszöget. Vége.</li>
</ol>

</body>
</html>

---


## 

**Pál Krisztián**  
Újvidéki Egyetem, Természettudományi-Matematikai Kar, hallgató

**Műhelyvezető**:  
Dr. Pintér Róbert

---



---
