
#../script/LongTest -m best.new.nn -b 12 -g 0 -i ~/NMT/corpus/03/03.cha.seg -o output/03.test.out

#nohup ../script/LongTest -m best.new.nn -b 12 -g 0 -i ~/NMT/corpus/03/03.cha.seg -o output/03.e.out &
#nohup ../script/LongTest -m best.new.nn -b 12 -g 0 -i ~/NMT/corpus/04/04.cha.seg -o output/04.e.out &
#nohup ../script/LongTest -m best.new.nn -b 12 -g 2 -i ~/NMT/corpus/05/05.cha.seg -o output/05.e.out &
#nohup ../script/LongTest -m best.new.nn -b 12 -g 2 -i ~/NMT/corpus/06/06.cha.seg -o output/06.e.out &

#../script/LongTest -m best.nn -v vocab.nn -b 12 -g 2 -i 03.seg -o 03.test

nohup ../script/LongTest -m best.nn -v vocab.nn -b 12 -g 3 -i ~/corpus/chn-tib/train/tib.dev.bpe -o output/tib.dev.bpe.out &
nohup ../script/LongTest -m best.nn -v vocab.nn -b 12 -g 1 -i ~/corpus/chn-tib/train/tib.test.bpe -o output/tib.test.bpe.out &

#nohup ../script/LongTest -m save_models_9.000000.nn -v vocab.nn -b 12 -g 0 -i ~/corpus/04/04.seg -o output/04.9.out &
#nohup ../script/LongTest -m save_models_9.000000.nn -v vocab.nn -b 12 -g 0 -i ~/corpus/05/05.seg -o output/05.9.out &
#nohup ../script/LongTest -m save_models_9.000000.nn -v vocab.nn -b 12 -g 0 -i ~/corpus/06/06.seg -o output/06.9.out &

#nohup ../script/LongTest -m best.15.nn -v vocab.nn -b 12 -g 3 -i ~/corpus/04/04.seg -o output/04.15.thre.out &
#nohup ../script/LongTest -m best.15.nn -v vocab.nn -b 12 -g 0 -i ~/corpus/05/05.seg -o output/05.15.thre.out &
#nohup ../script/LongTest -m best.15.nn -v vocab.nn -b 12 -g 1 -i ~/corpus/06/06.seg -o output/06.15.thre.out &
#nohup ../script/LongTest -m best.15.nn -v vocab.nn -b 12 -g 3 -i ~/corpus/08/08.seg -o output/08.15.thre.out &

#nohup ../script/LongTest -m best.nn -v vocab.nn -b 12 -g 0 -i ~/corpus/08/08.seg -o output/08.out.2 &
#../script/LongTest -m save_models_16.000000.nn -v vocab.nn -b 12 -g 0 -i ~/corpus/03/03.seg -o output/03.out
