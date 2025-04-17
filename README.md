![ek-logo-szoveggel](https://github.com/user-attachments/assets/b69ccf55-586c-4fc7-a8fc-c51f338b2076)



# Delaunay-háromszögelés Pythonban

Ez a projekt az **Európa Kollégium Informatika-műhelyének** keretén belül készült.

A projekt célja egy **Python nyelvű implementáció** készítése, amely **Delaunay-háromszögelést** végez a **Randomized Incremental Construction (RIC)** algoritmus segítségével.

##  Az algoritmus rövid ismertetése

<html lang="hu">
<head>
  <meta charset="UTF-8">
  <title>Delaunay Háromszögelés Algoritmus</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      line-height: 1.6;
      max-width: 800px;
      margin: 2rem auto;
      padding: 1rem;
      background-color: #f9f9f9;
      color: #333;
    }
    h2 {
      border-bottom: 2px solid #4a90e2;
      padding-bottom: 0.3rem;
      color: #4a90e2;
    }
    ol, ul {
      margin-left: 1.5rem;
    }
    code {
      background-color: #eef;
      padding: 0.2rem 0.4rem;
      border-radius: 3px;
      font-family: Consolas, monospace;
    }
    .step {
      margin: 0.5rem 0;
    }
  </style>
</head>
<body>
  <h2>INPUT</h2>
  <p>P pontokat tartalmazó halmaz, amely <code>n</code> darab (<code>x,y</code>) koordinátapárból áll.</p>

  <h2>OUTPUT</h2>
  <p>Egy <code>matplotlib</code> grafikon, amely megrajzolja az algoritmus futása során létrehozott háromszögeket.</p>

  <h2>ALGORITMUS LÉPÉSEI</h2>
  <ol>
    <li class="step">
      Létrehozzuk a külső szuperstruktúrát (<em>super-triangle</em>), amely elég nagy ahhoz, hogy ne zavarja a belső háromszögelést.
    </li>
    <li class="step">
      Véletlenszerű sorrendbe állítjuk a <code>P</code> ponthalmaz elemeit.
    </li>
    <li class="step">
      Egy fa struktúrát (DAG) hozunk létre, melynek gyökere a szuperstruktúra.
    </li>
    <li class="step">
      A pontokat egyesével beillesztjük:<ul>
        <li>Megvizsgáljuk <code>insideTriangle</code>-gal (baricentrikus koordináták):</li>
        <li>Ha a pont belül van (<code>True</code>):
          <ul>
            <li>Ha a node levél, beszúrjuk (<code>add_point_inside</code>), három új levél háromszöget létrehozva, majd <code>legalize_edge</code>.</li>
          </ul>
        </li>
        <li>Ha a pont egy élen van (<code>[v1,v2]</code>):
          <ul>
            <li>Megkeressük a szomszédos háromszöget (<code>find_adjacent_triangle</code>),</li>
            <li>Felosztjuk az élt, gyerekeket létrehozva, majd <code>legalize_edge</code>.</li>
          </ul>
        </li>
        <li>Ha nem levél, rekurzívan folytatjuk csak a tartalmazó gyereknél.</li>
      </ul>
    </li>
    <li class="step">
      Kirajzoljuk a fa összes levél háromszögét (<code>plot_leaves</code>), majd megjelenítjük a grafikont.
    </li>
  </ol>

  <p><strong>Megjegyzés:</strong> A CSS stílus letisztult, világos színekkel, könnyen olvasható betűtípussal és szerkezettel segíti a dokumentum áttekintését.</p>
</body>
</html


---


## 

**Pál Krisztián**  
Újvidéki Egyetem, Természettudományi-Matematikai Kar, hallgató

**Műhelyvezető**:  
Dr. Pintér Róbert

---



---
