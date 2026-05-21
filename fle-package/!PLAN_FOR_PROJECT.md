# Ogolny zarys zmian + plan projektu

## Plan:
Model RL będzie miał ograniczone zasoby, które w czasie rzeczywistym, będzie musiał rozstawić w odpowiednich miejscach na mapie (jezeli sie uda, to mozliwie, nieoznaczonch, ale w uproszczeniu do dla oznaczonych lepiej, mniej stanów dla wieżyczek) i postać będzie musiała dynamicznie zarządzać amunicją w zależności, czy na RADARZE (który będzie sercem bazy) pojawią się robale które, będą chciały mnie rozwalić
Prosty Naiwny Algorytm będzie zarządzał, gdzie robale zaczynają na mapie (to, skąd robale idą, niekoniecznie będzie modelowi wiadome, bo będzie miał tylko swoją część mapy z radarem)

W momencie, gdy do Modelu trafi C*Nty tick gry z mapą (C to mnożnik, by każdy tick gry nie dawać do modelu), model uczy się rozpoznania, gdzie w tym czasie są przeciwnicy.
W czasie zanim dostanie kolejny tick gry (albo w momencie na ile to możliwe), powinien być w stanie się w tym miejscu obronić przed Napastnikami

Celem modelu, jest przetrwanie pod atakami wroga przez dostatecznie najdłuższy możliwy czas (obrona to znaczy, że nie zginiemy, albo radar nie będzie zniszczony)

### Dostępne Stany modelu:
Model zna (docelowo):
- Mapę (obraz Pikselów mapy, w miare możliwości, czy wystarczy mu info z całego radaru)
- Ile ma Naboi w EQ
- Ile ma wieżyczek dostępnych, na jakich pozycjach już stoją w obronie
- Ile ma postawionych murów, gdzie są postawione (oznaczony tereny gdzie mogą być postawione)
- Mozliwe, ze pozycje wierzyczek rozpozna z pozycji na mapie

PS: Aby nie chodzić postacią, można spróbować wymusić tylko planowanie, a wszystko jest stawiane przez Roboty Konstrukcyjne w Factorio

### Dostępne akcje:
Model może:
- Postawić Mur w danym miejscu
- Postawić wieżyczkę w danym miejscu + automatyczne dodanie STACKA AMUNICJI (lub wybór ile amunicji dać)
- Przenieść mur z miejsca X do Y
- Przenieść wieżyczkę z miejsca X do Y
- Wypełnienie wieżyczki amunicją (moze, moze i nie, zalezy od complexity)
- Strzelanie z własnego karabinu

(AMUNICJA RACZEJ BĘDZIE PERIODYCZNIE DODAWANA DO EKWIPUNKU, ALE ZEBY NiE WYPELNIC ZA BARDZO)

1. Wykorzystanie FactorioInstance do odpalenia gry w instancji obrazu dockerowego

Wgranie instancji Factorio z:
- minimalną mapą stanów 
(
ma wgrane swój teren startowy, półbezpieczny
kilka spawnerów robali, na średnim poziomie agresji, ale agresja wzsrasta wraz z długością gry
inventory zawierające podstawowe narzędzia obronne
    - Pistolet/Karabin (początkowe narzędzie obronne, ale wraz z czasem w funkcji celu powinno być wyparte)
    - Naboje (mozliwe, ze nieskonczone)
    - Wieżyczki (podstawowe), lecz nie za dużo by nauczyć się pokrycia terenu w momencie gdy robale przychodzą
    - Mury (nie za dużo, tak jak powyżej)
    )
- inventory z podstawowymi 
2. Uproszczenie FactorioGymEnvironment do przyjmowania instancji modelu głębokiego RL
- najpierw, określenie dokładnych stanów modelu. Model RL powinien mieć kilka akcji