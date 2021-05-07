mkdir -p data/udpipe/$1/5fold

# ./udpipe --train --tokenizer=none --parser=none data/udpipe/$1/5fold/1.model data/conllu-folds/$1/fold-2.conllu data/conllu-folds/$1/fold-3.conllu data/conllu-folds/$1/fold-4.conllu data/conllu-folds/$1/fold-5.conllu
# ./udpipe --train --tokenizer=none --parser=none data/udpipe/$1/5fold/2.model data/conllu-folds/$1/fold-1.conllu data/conllu-folds/$1/fold-3.conllu data/conllu-folds/$1/fold-4.conllu data/conllu-folds/$1/fold-5.conllu
# ./udpipe --train --tokenizer=none --parser=none data/udpipe/$1/5fold/3.model data/conllu-folds/$1/fold-1.conllu data/conllu-folds/$1/fold-2.conllu data/conllu-folds/$1/fold-4.conllu data/conllu-folds/$1/fold-5.conllu
# ./udpipe --train --tokenizer=none --parser=none data/udpipe/$1/5fold/4.model data/conllu-folds/$1/fold-1.conllu data/conllu-folds/$1/fold-2.conllu data/conllu-folds/$1/fold-3.conllu data/conllu-folds/$1/fold-5.conllu
# ./udpipe --train --tokenizer=none --parser=none data/udpipe/$1/5fold/5.model data/conllu-folds/$1/fold-1.conllu data/conllu-folds/$1/fold-2.conllu data/conllu-folds/$1/fold-3.conllu data/conllu-folds/$1/fold-4.conllu

echo
for i in 1 2 3 4 5; do
    ./udpipe --tag --accuracy data/udpipe/$1/5fold/$i.model data/conllu-folds/$1/fold-$i.conllu
done


# frisian:
# Tagging from gold tokenization - forms: 3025, upostag: 90.71%, xpostag: 78.88%, feats: 80.56%, alltags: 63.14%, lemmas: 80.66%
# Tagging from gold tokenization - forms: 3187, upostag: 90.15%, xpostag: 78.82%, feats: 80.26%, alltags: 61.72%, lemmas: 80.92%
# Tagging from gold tokenization - forms: 3170, upostag: 90.00%, xpostag: 77.32%, feats: 79.56%, alltags: 61.01%, lemmas: 82.84%
# Tagging from gold tokenization - forms: 3255, upostag: 91.64%, xpostag: 80.58%, feats: 82.70%, alltags: 66.24%, lemmas: 81.72%
# Tagging from gold tokenization - forms: 3232, upostag: 90.50%, xpostag: 76.79%, feats: 79.46%, alltags: 61.01%, lemmas: 83.23%

# (90.71 + 90.15 + 90.00 + 91.64 + 90.50) / 5 = 90.60
# std: $\sigma = 0.58$


# groningen:
# Tagging from gold tokenization - forms: 7555, upostag: 91.04%, xpostag: 100.00%, feats: 100.00%, alltags: 91.04%, lemmas: 87.77%
# Tagging from gold tokenization - forms: 5081, upostag: 93.15%, xpostag: 100.00%, feats: 100.00%, alltags: 93.15%, lemmas: 88.88%
# Tagging from gold tokenization - forms: 11112, upostag: 91.91%, xpostag: 100.00%, feats: 100.00%, alltags: 91.91%, lemmas: 87.80%
# Tagging from gold tokenization - forms: 9419, upostag: 92.21%, xpostag: 100.00%, feats: 100.00%, alltags: 92.21%, lemmas: 88.28%
# Tagging from gold tokenization - forms: 15888, upostag: 90.94%, xpostag: 100.00%, feats: 100.00%, alltags: 90.94%, lemmas: 85.79%

# (91.04 + 93.15 + 91.91 + 92.21 + 90.94) / 5 = 91.85
# std: $\sigma = 0.81$


# We use UDPipe \citep{straka-etal-2016-udpipe} with five-fold cross-validation on the labeled target data as a baseline method for evaluating POS-tagging performance.
# UDPipe achieves an average accuracy of 91.85 ($\sigma = 0.81$) for Gronings and 90.60 ($\sigma = 0.58$) for Frisian.
# These accuracies are upper limits, since heldout data is sampled from the training data.
# When the Frisian data is split in three folds corresponding to the data sources, a lower cross-source average accuracy of 87.37 ($\sigma = 2.88$) is achieved.
