# Arguments: VERSION
# VERSION: file in configs/mlm

for f in configs/mlm/data/*.json; do
    fname="$(basename $f)"
    name="${fname%.*}"
    echo $name
    echo $ ts python -m src.train.train_mlm configs/mlm/$1.json $f
    if [ ! -d "data/output/mlm/$1/$name" ]; then
        echo ts python -m src.train.train_mlm configs/mlm/$1.json $f
    else
        echo skipping
    fi
    echo
done
