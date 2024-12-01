while true; do
    python aboutpet.item2vec.prep.py
    sleep 3
    python aboutpet.popular.py
    sleep 3
    python aboutpet.item2vec.delete.py
    sleep 3
    python aboutpet.item2vec.train.py
    sleep 3
    python aboutpet.item2vec.update.py
    sleep 3
done
