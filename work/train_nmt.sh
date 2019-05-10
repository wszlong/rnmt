#!/bin/bash

#nohup ../script/LongTrain -t ~/NMT/corpus/big.cha.ch ~/NMT/corpus/big.en model.cha.nn -a ~/NMT/corpus/03/03.cha.seg ~/NMT/corpus/03/ref.0 -B best.cha.nn -n 20 -d 0.8 -V 30000 -v 30000 -m 128 -N 2 -M 1 1 1 -E 1000 -H 1000 --feed-input true --tmp-dir-location tmp > logfile.cha &

#nohup ../script/LongTrain -C ~/NMT/corpus/big.cha.ch ~/NMT/corpus/big.en best.cha.nn -a ~/NMT/corpus/03/03.cha.seg ~/NMT/corpus/03/ref.0 -B best.new.nn -n 20 -d 0.8 -l 0.025 -V 30000 -v 30000 -m 128 -N 2 -M 1 1 1 -E 1000 -H 1000 --feed-input true --tmp-dir-location tmp > logfile.cha.new &

nohup ../script/LongTrain -t ~/corpus/chn-tib/train/tib.train.bpe ~/corpus/chn-tib/train/chn.train.bpe vocab.nn model.nn -a ~/corpus/chn-tib/train/tib.dev.bpe ~/corpus/chn-tib/train/chn.dev.bpe -B best.nn --save-after-n-epoch 3 -n 10 -d 0.8 --adam 0.001 4 -l 0.1 -V 36000 -v 31000 -m 128 -N 2 -M 1 0 3 -E 1000 -H 1000 --feed-input true --tmp-dir-location tmp > logfile &
