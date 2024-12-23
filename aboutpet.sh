while true; do
    python aboutpet.item2vec.v1.prep.py
    sleep 3
    python aboutpet.popular.py
    sleep 3
    python aboutpet.item2vec.v1.delete.py
    sleep 3
    python aboutpet.item2vec.v1.train.py
    sleep 3
    python aboutpet.item2vec.v1.export.py
    sleep 3
    python aboutpet.carca.v1.train.py
    sleep 3
    python aboutpet.carca.v1.export.py
    sleep 3
done
