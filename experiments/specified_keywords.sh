#!/bin/sh
cd ..

cd first-sentence-generator/
./specified_keywords.sh
cd ..

cd rnnpg-generator
./specified_keywords.sh
cd ..

echo "checkout the generated poems at ./keywords.poems.std"
