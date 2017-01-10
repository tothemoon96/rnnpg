#!/bin/sh

keywordsF=../MISC/rand.keywords.30
outputF=$keywordsF.poems.out
outstdF=$keywordsF.poems.std

codedir=rnn_poem_postprecessing
cd $codedir
if [ ! -d "./bin" ] ; then
	mkdir bin
fi
if [ ! -d "./dist" ] ; then
	mkdir dist
fi
javac -sourcepath ./src -d ./bin ./src/dio/*.java
javac -sourcepath ./src -d ./bin ./src/rnn_poem/*.java
jar -cfe dist/rnn_poem_post_precessor.jar rnn_poem.Postprocessor -C bin/ .
cd ..
cp $codedir/dist/rnn_poem_post_precessor.jar rnn_poem_post_precessor.jar

./rnnpg-generator poem_generator.conf $keywordsF.firstSent.txt $outputF

# convert the verbose format to human readable format
java -jar rnn_poem_post_precessor.jar $outputF $outstdF
