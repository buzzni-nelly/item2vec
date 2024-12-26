while true; do
    python exec.item2vec.prep.py --company-id aboutpet --version v1
    sleep 3
    python aboutpet.popular.export.py
    sleep 3
    python exec.item2vec.delete.py --company-id aboutpet --version v1
    sleep 3
    python exec.item2vec.train.py --company-id aboutpet --version v1
    sleep 3
    python exec.item2vec.export.py --company-id aboutpet --version v1
    sleep 3
    python exec.carca.delete.py --company-id aboutpet --version v1
    sleep 3
    python exec.carca.train.py --company-id aboutpet --version v1
    sleep 3
    python exec.carca.export.py --company-id aboutpet --version v1
    sleep 3
    python exec.carca.onnx.validation.py --company-id aboutpet --version v1
    sleep 3
done
