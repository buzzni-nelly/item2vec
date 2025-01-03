while true; do
    python exec.item2vec.prep.py --company-id wshop --version v1 --trace-begin-date 20240801
    sleep 3
    python exec.item2vec.delete.py --company-id wshop --version v1
    sleep 3
    python exec.item2vec.train.py --company-id wshop --version v1
    sleep 3
    python exec.item2vec.export.py --company-id wshop --version v1
    sleep 3
    python exec.carca.delete.py --company-id wshop --version v1
    sleep 3
    python exec.carca.train.py --company-id wshop --version v1
    sleep 3
    python exec.carca.export.py --company-id wshop --version v1
    sleep 3
    python exec.carca.onnx.validation.py --company-id wshop --version v1
    sleep 3
done
