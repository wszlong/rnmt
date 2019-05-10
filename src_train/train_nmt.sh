#!/bin/bash

#nohup ../script/LongTrain -t ~/NMT/corpus/big.cha.ch ~/NMT/corpus/big.en model.cha.nn -a ~/NMT/corpus/03/03.cha.seg ~/NMT/corpus/03/ref.0 -B best.cha.nn -n 20 -d 0.8 -V 30000 -v 30000 -m 128 -N 2 -M 1 1 1 -E 1000 -H 1000 --feed-input true --tmp-dir-location tmp > logfile.cha &

#nohup ../script/LongTrain -C ~/NMT/corpus/big.cha.ch ~/NMT/corpus/big.en best.cha.nn -a ~/NMT/corpus/03/03.cha.seg ~/NMT/corpus/03/ref.0 -B best.new.nn -n 20 -d 0.8 -l 0.025 -V 30000 -v 30000 -m 128 -N 2 -M 1 1 1 -E 1000 -H 1000 --feed-input true --tmp-dir-location tmp > logfile.cha.new &

#nohup ../script/LongTrain -C ~/corpus/small.ch ~/corpus/small.en vocab.nn model.nn -a ~/corpus/03/03.seg ~/corpus/03/ref.0 -B best.nn -n 20 -d 0.8 -l 0.1 -V 30000 -v 30000 -m 128 -N 2 -M 0 2 3 -E 1000 -H 1000 --feed-input true --tmp-dir-location tmp > logfile &

./LongTrain -t ~/corpus/small.ch ~/corpus/small.en vocab.nn model.nn -a ~/corpus/03/03.seg ~/corpus/03/ref.0 -B best.nn --save-after-n-epoch 10 -n 20 -d 0.8 --adam 0.001 4 -l 0.1 -V 3000 -v 3000 -m 32 -N 2 -M 2 1 0 -E 100 -H 100 --feed-input true --tmp-dir-location tmp  

#./LongTrain -t ~/corpus/small.ch ~/corpus/small.en vocab.nn model.nn -a ~/corpus/03/03.seg ~/corpus/03/ref.0 -B best.nn --save-after-n-epoch 10 -n 20  -l 0.1 -V 3000 -v 3000 -m 32 -N 3 -M 1 0 2 3 -E 100 -H 100 --feed-input true --tmp-dir-location tmp  
