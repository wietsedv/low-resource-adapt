# Arguments: VERSION
# VERSION: file in configs/pos

for f in configs/pos/data/*.json; do
    fname="$(basename $f)"
    name="${fname%.*}"
    echo $name
    echo $ ts python -m src.train.train_pos configs/pos/$1.json $f
    if [ ! -d "data/output/pos/$1/$name" ]; then
        ts python -m src.train.train_pos configs/pos/$1.json $f
    else
        echo skipping
    fi
    echo
done
