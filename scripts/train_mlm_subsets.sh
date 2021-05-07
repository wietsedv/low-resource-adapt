# Arguments: VERSION
# VERSION: file in configs/mlm


m=${2:-}

for f in configs/mlm/data/$m*.json; do
    fname="$(basename $f)"
    name="${fname%.*}"
    lang="$(echo $name | cut -d'_' -f3)"

    for f2 in configs/mlm/subsets/${lang}_*.json; do
        fname="$(basename $f2)"
        sname="$name-$(echo ${fname%.*} | cut -d'_' -f2)"

        # echo ts -L $sname python -m src.train.train_mlm configs/mlm/$1.json $f $f2
        echo sbatch -J $sname mlm_job.sh configs/mlm/$1.json $f $f2

        # if [ ! -d "data/output/mlm/$1/$sname" ]; then
        #     ts -L $sname python -m src.train.train_mlm configs/mlm/$1.json $f $f2
        #     # echo sbatch -J $sname mlm_job.sh configs/mlm/$1.json $f $f2
        # else
        #     echo skipping
        # fi
        # echo
    done
done
