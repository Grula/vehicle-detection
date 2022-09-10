Ovaj paket sadrži stil doktorata Matematičkog fakulteta, Univerziteta u
Beogradu. Stil je kreiran na osnovu uputstva Univerziteta u Beogradu.

**Autor**: Filip Marić
**Verzija**: 1.0
**Datum**: Oktobar 2015.

# Osnovne pretpostavke

Da bi paket mogao da se koristi, potrebno je da je na računaru
instaliran neki od novijiih LaTeX sistema (MikTeX ili TeXLive). Opis
instalacije ovih sistema i potrebnih fontova opisan je u datoteci
`INSTALL`. Paket se može koristiti u kombinaciji sa bilo kojim
editorom i integrisanim razvojnim okruženjem za LaTeX koji podržavaju
UNICODE i dopuštaju unos ćiriličkog teksta i latiničkih dijakritika.

Paket se može koristiti iz dva LaTeX prevodioca:

  - `pdflatex` - klasična, najrasprostranjenija varijanta koja koristi
    specijalne fontove za LaTeX i nema podršku za pun skup UNICODE
    karaktera. Najveći problem za srpski jezik su mala ćirilička slova
    koja se u italic varijanti ne prikazuju ispravno.
    
  - `xelatex` - novija verzija prevodioca za LaTeX koja u potpunosti
     podržava pun skup UNICODE karaktera i koristi sistemske fontove
     (OpenType i TrueType). Kada se koristi ovaj prevodilac, dobija se
     PDF koji sadrži potpuno ispravno predstavljena sva slova,
     uključujući i mogućnost pretrage PDF dokumenta (uključujući i
     ispravno indeksiranje od strane pretraživačkih mašina na vebu),
     kopiranja njegovih delova u druge dokumente.

Oba prevodioca su instalirana prilikom instalacije LaTeX distribucije
(izbor između njih se obično veoma jednostavno vrši u
editoru). Preporuka autora paketa je da se koristi prevodilac
`xelatex`, naročito ako se teza piše ćiriličkim pismom.

# Sadržaj paketa

Meta informacije:

  - `README`  - ova datoteka
  - `INSTALL` - uputstva za instalaciju LaTeX sistema, pratećih paketa i fontova

Datoteke koje implementiraju stil doktorata:

  - `matfdoktorat/matfdoktorat.sty` - latex paket koji definiše sve potrebne parametre
  - `matfdoktorat/_Prilog1.pdf`     - izjava o autorstvu koja se uključuje u doktorat
  - `matfdoktorat/_Prilog2.pdf`     - izjava o istovetnosti štampane i elektronske verzije
                                          koja se uključuje u doktorat
  - `matfdoktorat/_Prilog3.pdf`     - izjava o korišćenju koja se uključuje u doktorat
  - `matfdoktorat/hyperref.cfg`     - podešavanja paketa hyperref (obezbeđuje linkove u PDF-u)
  - `matfdoktorat/serbianc.ldf`     - podešavanje paketa babel za srpski jezik, ćirilicom
                                          (malo izmenjeno u odnosu na originalno)
  - `matfdoktorat/serbianc.lbx`     - podešavanje paketa biblatex za srpski jezik, ćirilicom 
  - `matfdoktorat/serbian.lbx`      - podešavanje paketa biblatex za srpski jezik, latiniciom

Datoteke koje prikazuju korišćenje stila:

  - `matfdoktorat/matfdoktorat-primer.tex`        - primer ćiriličkog doktorata
  - `matfdoktorat/matfdoktorat-primer-lat.tex`    - primer latiničkog doktorata
  - `matfdoktorat/matfdoktorat-primer-cirlat.tex`    - primer ćiriličkog doktorata kucanog latiniciom
  - `matfdoktorat/matfdoktorat-primer.bib`        - bibliografija (bibtex) koja se koristi u primeru
  - `matfdoktorat/pangrami.sty`                   - pomoćni paket koji sadrži nasumični tekst
                                                        (pangrame) na raznim jezicima
  - `matfdoktorat/graph.png`                      - slika koja se uključuje u primeru

Datoteke koje implementiraju stil master rada:

  - `matfmaster/matfmaster.sty` - latex paket koji definiše sve potrebne parametre
  - `matfmaster/hyperref.cfg`     - podešavanja paketa hyperref (obezbeđuje linkove u PDF-u)
  - `matfmaster/serbianc.ldf`     - podešavanje paketa babel za srpski jezik, ćirilicom
                                        (malo izmenjeno u odnosu na originalno)
  - `matfmaster/serbianc.lbx`     - podešavanje paketa biblatex za srpski jezik, ćirilicom 
  - `matfmaster/serbian.lbx`      - podešavanje paketa biblatex za srpski jezik, latiniciom

Datoteke koje prikazuju korišćenje stila:

  - `matfmaster/matfmaster-primer.tex`         - primer ćiriličkog mastera
  - `matfmaster/matfmaster-primer-lat.tex`    - primer latiničkog mastera
  - `matfmaster/matfmaster-primer.bib`         - bibliografija (bibtex) koja se koristi u primeru
  - `matfmaster/pangrami.sty`                   - pomoćni paket koji sadrži nasumični tekst
                                                (pangrame) na raznim jezicima
  - `matfmaster/graph.png`                      - slika koja se uključuje u primeru

Fontovi za XeLaTeX:

  - `fonts/CMU/*`        - Computer Modern Unicode (pod Linux-om je bolje instalirati ih iz paketa fonts-cmu)
  - `fonts/OpenSans/*`   - Open Sans

Uputstva Univerziteta u Beogradu:

  - `univerzitet/Forma_doktorske_disertacije.pdf`        - kratak opis forme doktorske disertacije
  - `univerzitet/uputstvo-za-doktorske-disertacije.pdf`  - duži opis sadržaja i forme doktorske disertacije



## Postupak prevođenja 

Primer matfdoktorat-primer se prevodi na sledeći način
(veoma slično se postupa i sa matfdoktorat-primer-lat):

Ako se koristi pdflatex i bibtex:

~~~
    pdflatex matfdoktorat-primer
    bibtex matfdoktorat-primer
    pdflatex matfdoktorat-primer
    pdflatex matfdoktorat-primer
~~~

  Ako se koristi pdflatex i biblatex/biber:
    obavezno navesti opciju biblatex tj. uključiti paket matfdoktorat
    pomoću `\usepackage[biblatex]{matfdoktorat}`
    
~~~
    pdflatex matfdoktorat-primer
    biber matfdoktorat-primer
    pdflatex matfdoktorat-primer
    pdflatex matfdoktorat-primer
~~~

  Ako se koristi xelatex i bibtex:

~~~
    xelatex matfdoktorat-primer
    bibtex matfdoktorat-primer
    xelatex matfdoktorat-primer
    xelatex matfdoktorat-primer
~~~

  Ako se koristi xelatex i biblatex/biber:
    obavezno navesti opciju biblatex tj. uključiti paket matfdoktorat
    pomoću `\usepackage[biblatex]{matfdoktorat}`
    
~~~
    xelatex matfdoktorat-primer
    biber matfdoktorat-primer
    xelatex matfdoktorat-primer
    xelatex matfdoktorat-primer
~~~

Sve ovo je olakšano ako se koristi Make sistem za latex (latexmk). Na primer:

~~~
   latexmk -pdf matfdoktorat-primer   - kompletno prevodi matfdoktorat-primer
   latexmk -c matfdoktorat-primer     - čisti sve pomoćne datoteke nastale
                                        prevođenjem matfdoktorat-primer
~~~
