# Chinese Poetry Generation with Recurrent Neural Networks
This project includes the code/model for the paper 

[Chinese Poetry Generation with Recurrent Neural Networks](http://aclweb.org/anthology/D/D14/D14-1074.pdf)


```
@InProceedings{zhang-lapata:2014:EMNLP2014,
  author    = {Zhang, Xingxing  and  Lapata, Mirella},
  title     = {Chinese Poetry Generation with Recurrent Neural Networks},
  booktitle = {Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  month     = {October},
  year      = {2014},
  address   = {Doha, Qatar},
  publisher = {Association for Computational Linguistics},
  pages     = {670--680},
  url       = {http://www.aclweb.org/anthology/D14-1074}
}
```

# Acknowledgement
Our implementation is greatly inspired by Tomas Mikolov's [rnnlm toolkit](http://rnnlm.org/).
We would like to thank Tomas Mikolov for making his code public available.

# Dataset
Download the complete dataset from [here](http://homepages.inf.ed.ac.uk/mlap/Data/EMNLP14/)

# Dependencies
* [KenLM](https://github.com/XingxingZhang/rnnpg/tree/master)
* G++ (4.4.7)
* Java (1.8.0_51, 1.6 or 1.7 should also be fine)
* Python (2.7)

# Installation
1) Install [KenLM](https://github.com/XingxingZhang/rnnpg/tree/master)，并将[KenLM](https://github.com/XingxingZhang/rnnpg/tree/master)回退到合适的版本：
```
git reset --hard C090f0bff25d0761bdcf9c1700e37f898c3c029c
```
Also remember to add kenlm to your KENLM_PATH
```
# 你的kenlm的目录，例如 
export KENLM_PATH=/home/tothemoon/Project/rnnpg/kenlm
```
2) 在项目根目录下执行make

+ Debug版：
```
make -j4 debug
```
+ Release版：
```
make -j4 release
```
3) 执行清理工作
```
make clean
```



# Run Experiments
Download data/model from [here](https://drive.google.com/file/d/0B6-YKFW-MnbOYnJDeWVXRnlObzA/view?usp=sharing)
```
# move MISC.tar.bz2 to the root folder of this project, then
tar jxvf MISC.tar.bz2
```
**1. Perplexity**
```
cd experiments
./ppl.sh
```

**2. Generation**
```
cd experiments
./generation.sh
```
Enjoy the generated poems!

**3. BLEU**

Download from [here](https://drive.google.com/file/d/0B6-YKFW-MnbORk16WmNXbDhsVk0/view?usp=sharing)
```
tar jxvf BLEU2-final.tar.bz2 
cd BLEU2-final
cd MERT_channel-1_RNN-CB-POS-LM-Eval-BLEU2
python showBLEU.py .
```
