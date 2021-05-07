rm fy_testap.model
rm fy_nieuws.model
rm fy_tresoar.model

./udpipe --train --tokenizer=none --parser=none fy_testap.model data/sources/ud-frisian/nieuws-combined.conllu data/sources/ud-frisian/tresoar.conllu
./udpipe --train --tokenizer=none --parser=none fy_nieuws.model data/sources/ud-frisian/fy_testap.conllu data/sources/ud-frisian/tresoar.conllu
./udpipe --train --tokenizer=none --parser=none fy_tresoar.model data/sources/ud-frisian/fy_testap.conllu data/sources/ud-frisian/nieuws-combined.conllu

# echo
# for m in fy_*.model; do ./udpipe --tag --accuracy $m data/sources/ud-frisian/*.conllu; done

./udpipe --tag --accuracy fy_testap.model data/sources/ud-frisian/fy_testap.conllu
./udpipe --tag --accuracy fy_nieuws.model data/sources/ud-frisian/nieuws-combined.conllu
./udpipe --tag --accuracy fy_tresoar.model data/sources/ud-frisian/tresoar.conllu

# Loading UDPipe model: done.
# Tagging from gold tokenization - forms: 1618, upostag: 86.59%, xpostag: 0.00%, feats: 80.96%, alltags: 0.00%, lemmas: 31.09%
# Tagging from gold tokenization - forms: 4595, upostag: 99.98%, xpostag: 99.98%, feats: 100.00%, alltags: 99.98%, lemmas: 99.96%
# Tagging from gold tokenization - forms: 9656, upostag: 82.58%, xpostag: 0.02%, feats: 71.17%, alltags: 0.01%, lemmas: 86.72%
# Loading UDPipe model: done.
# Tagging from gold tokenization - forms: 1618, upostag: 100.00%, xpostag: 100.00%, feats: 100.00%, alltags: 100.00%, lemmas: 99.44%
# Tagging from gold tokenization - forms: 4595, upostag: 74.93%, xpostag: 0.02%, feats: 70.58%, alltags: 0.02%, lemmas: 36.97%
# Tagging from gold tokenization - forms: 9656, upostag: 75.48%, xpostag: 99.99%, feats: 65.67%, alltags: 61.12%, lemmas: 42.15%
# Loading UDPipe model: done.
# Tagging from gold tokenization - forms: 1618, upostag: 86.46%, xpostag: 100.00%, feats: 78.55%, alltags: 75.83%, lemmas: 30.78%
# Tagging from gold tokenization - forms: 4595, upostag: 84.48%, xpostag: 0.02%, feats: 75.56%, alltags: 0.02%, lemmas: 89.14%
# Tagging from gold tokenization - forms: 9656, upostag: 99.96%, xpostag: 100.00%, feats: 99.89%, alltags: 99.87%, lemmas: 99.68%

# ((86.59 + 82.58) + (74.93 + 75.48) + (86.46 + 84.48)) / 6 = 81.75
# std: $\sigma = 4.82$


# Tagging from gold tokenization - forms: 1618, upostag: 90.05%, xpostag: 72.74%, feats: 81.71%, alltags: 57.11%, lemmas: 30.96%
# Tagging from gold tokenization - forms: 4595, upostag: 88.68%, xpostag: 0.02%, feats: 79.43%, alltags: 0.02%, lemmas: 78.41%
# Tagging from gold tokenization - forms: 9656, upostag: 83.38%, xpostag: 10.16%, feats: 72.14%, alltags: 6.95%, lemmas: 82.55%

# (90.05 + 88.68 + 83.38) = 87.37
# std: $\sigma = 2.88$