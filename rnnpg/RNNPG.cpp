/*
 * RNNPG.cpp
 *
 *  Created on: 27 Dec 2013
 *      Author: s1270921
 */

#include <iostream>
#include <cstdio>
#include <cassert>
using namespace std;

#include "RNNPG.h"
#include "xutil.h"

//#ifndef M_LN10
//#define M_LN10 2.30258509299404568402  /* log_e 10 */
//#endif
//
//double exp10 (double arg) { return exp(M_LN10 * arg); }

RNNPG::RNNPG()
{
	hiddenSize = 200;
	classSize = 100;
	trainFile[0] = 0;
	validFile[0] = 0;
	testFile[0] = 0;
	vocabClassF[0] = 0;
	//这里好像包含了一些随机性，看看是在哪里
	randomSeed = 1;
	srand(randomSeed);

	wordEmbeddingFile[0] = 0;

	int i;
	for(i = 0; i < MAX_CON_N; i ++)
	{
		conSyn[i] = NULL;
		conSynOffset[i] = NULL;
	}

	firstTimeInit = true;
	maxIter = 15;
	for(i = 0; i < SEN5_HIGHT; i ++)
		sen5Neu[i] = NULL;
	for(i = 0; i < SEN7_HIGHT; i ++)
		sen7Neu[i] = NULL;

	compressSyn = NULL;
	hisNeu = NULL;
	cmbNeu = NULL;

	for(i = 0; i < 8; i ++)
		map7Syn[i] = NULL;
	for(i = 0; i < 6; i ++)
		map5Syn[i] = NULL;
	conditionNeu = NULL;

	inNeu = NULL;
	hiddenNeu = NULL;
	outNeu = NULL;
	hiddenInSyn = NULL;
	outHiddenSyn = NULL;

	voc_arr = NULL;
	classStart = NULL;
	classEnd = NULL;

	alpha=0.1;
	alphaDiv = 2.0;
	beta=0.0000001;

	// for temp information
	wordCounter = 0;

	bpttHistory = NULL;
	bpttHiddenNeu = NULL;
	bpttInHiddenNeu = NULL;
	bpttHiddenInSyn = NULL;
	bpttConditionNeu = NULL;

	fixSentenceModelFirstLayer = false;
	logp = 0;
	totalPoemCount = 0;

	senweSyn = NULL;

	randomlyInitSenModelEmbedding = false;
	alphaDivide = 0;
	minImprovement = 1.0001; // 1.0001

	for(i = 0; i < MAX_CON_N; i ++)
		conSyn_backup[i] = NULL;
	for(i = 0; i < MAX_CON_N; i ++)
		conSynOffset_backup[i] = NULL;
	for(i = 0; i < SEN7_HIGHT; i ++)
		sen7Neu_backup[i] = NULL;
	for(i = 0; i < SEN5_HIGHT; i ++)
		sen5Neu_backup[i] = NULL;
	compressSyn_backup = NULL;
	hisNeu_backup = NULL;
	cmbNeu_backup = NULL;
	for(i = 0; i < 8; i ++)
		map7Syn_backup[i] = NULL;
	for(i = 0; i < 6; i ++)
		map5Syn_backup[i] = NULL;
	conditionNeu_backup = NULL;
	senweSyn_backup = NULL;
	inNeu_backup = NULL;
	hiddenNeu_backup = NULL;
	outNeu_backup = NULL;
	hiddenInSyn_backup = NULL;
	outHiddenSyn_backup = NULL;

	modelFile[0] = 0;
	saveModel = 0;

	stableAC = 0.1;
	historyStableAC = 0;

	flushOption = 1;

	consynMin = -10;
	consynMax = 10;
	consynOffset = 5;

	mode = UNDEFINED_MODE;

	outConditionDSyn = NULL;
	bufOutConditionNeu = NULL;
	outConditionDSyn_backup = NULL;
	// bufOutConditionNeu_backup = NULL;

	directError = false;
	conbptt = false;

	isLastSentOfPoem = false;

	conBPTTHis = NULL;
	conBPTTCmbHis = NULL;
	conBPTTCmbSent = NULL;

	contextBPTTSentNum = -1;

	bpttHisCmbSyn = NULL;

	perSentUpdate = false;

	adaGrad = false;

	adaGradEps = 1e-3;
}

RNNPG::~RNNPG()
{
	int i;
	for(i = 0; i < MAX_CON_N; i ++)
	{
		free(conSyn[i]);
		free(conSynOffset[i]);
	}
	for(i = 0; i < SEN5_HIGHT; i ++)
			free(sen5Neu[i]);
	for(i = 0; i < SEN7_HIGHT; i ++)
		free(sen7Neu[i]);
	free(compressSyn);
	free(hisNeu);
	free(cmbNeu);
	for(i = 0; i < 8; i ++)
		free(map7Syn[i]);
	for(i = 0; i < 6; i ++)
		free(map5Syn[i]);
	free(conditionNeu);
	free(inNeu);
	free(hiddenNeu);
	free(outNeu);
	free(hiddenInSyn);
	free(outHiddenSyn);

	free(classStart);
	free(classEnd);

	free(bpttHistory);
	free(bpttHiddenNeu);
	free(bpttInHiddenNeu);
	free(bpttHiddenInSyn);
	free(bpttConditionNeu);

	free(senweSyn);

	for(i = 0; i < MAX_CON_N; i ++)
		free(conSyn_backup[i]);
	for(i = 0; i < MAX_CON_N; i ++)
		free(conSynOffset_backup[i]);
	for(i = 0; i < SEN7_HIGHT; i ++)
		free(sen7Neu_backup[i]);
	for(i = 0; i < SEN5_HIGHT; i ++)
		free(sen5Neu_backup[i]);
	free(compressSyn_backup);
	free(hisNeu_backup);
	free(cmbNeu_backup);
	for(i = 0; i < 8; i ++)
		free(map7Syn_backup[i]);
	for(i = 0; i < 6; i ++)
		free(map5Syn_backup[i]);
	free(conditionNeu_backup);
	free(senweSyn_backup);
	free(inNeu_backup);
	free(hiddenNeu_backup);
	free(outNeu_backup);
	free(hiddenInSyn_backup);
	free(outHiddenSyn_backup);

	free(outConditionDSyn);
	free(bufOutConditionNeu);
	free(outConditionDSyn_backup);
	// free(bufOutConditionNeu_backup);

	free(conBPTTHis);
	free(conBPTTCmbHis);
	free(conBPTTCmbSent);

	free(bpttHisCmbSyn);

	if(adaGrad)
		sumGradSquare.releaseMemory();
}

/**
 * @brief
 * 初始化rnnpg,分配内存空间，设置一些初值，设置字符表
 */
void RNNPG::initNet()
{
	if(vocab.getVocabSize() == 0)
	{
		cout << "in initNet, vocabulary size = 0!!!" << endl;
		assert(vocab.getVocabSize() != 0);
	}

	int i, j, N;

	// =========== this is all for the sentence model ==============
	// allocate memory and initilize conSyn (the convolution matrix)
	for(i = 0; i < MAX_CON_N; i ++)
	{
		//初始化每一个卷积层C^{l,n}_{:.i}的权重和偏置，hiddenSize表示的是隐含层和Word Embedding词向量的维数，也就是说两个的维数要一致
		conSyn[i] = (synapse*)xmalloc(hiddenSize * conSeq[i] * sizeof(synapse), "in initNet (con syn)");
		//它的初值还没有初始化
		conSynOffset[i] = (synapse*)xmalloc(hiddenSize * conSeq[i] * sizeof(synapse), "in initNet (con syn offset)");
		N = hiddenSize * conSeq[i];//卷积层参数的数目
		//接下来进行权重的初始化
		for(j = 0; j < N; j ++)
		{
			// conSyn[i][j].weight = random(-0.1, 0.1)+random(-0.1, 0.1)+random(-0.1, 0.1);
			// it is important to initilize the matrix with large values
			// conSyn[i][j].weight = random(-10.0, 10.0);   // works much better than random(-0.1, 0.1)+random(-0.1, 0.1)+random(-0.1, 0.1);
			// now try this one
			// I am not sure if this is a good idea, but I will keep it first
			if(consynMin == -0.1 && consynMax == 0.1)
			{
				conSyn[i][j].weight = random(-0.1, 0.1)+random(-0.1, 0.1)+random(-0.1, 0.1);
			}
			else
			{
				double rnd = random(consynMin, consynMax);
				if(rnd < 0)
					rnd -= consynOffset;
				else
					rnd += consynOffset;
				conSyn[i][j].weight = rnd;
			}
		}
	}

	// load word embedding
	if(mode == TRAIN_MODE)
	{
		wdEmbed.load(wordEmbeddingFile);
		cout << "load word embedding done!" << endl;
	}

	// allocate memory and initilize sen5/7Neu (the sentence model)
	/* 还是在做卷积网络CSM的T^{l+1}_{:,j}初始化，分别创建对于5言诗和7言诗的每一层内部的神经元
	 * 对于5言诗，每一层分别为(5,4,3,1)*hiddenSize个神经元
	 * 对于7言诗，每一层分别为(7,6,5,3,1)*hiddenSize个神经元
	 */
	int unitNum = 5;
	for(i = 0; i < SEN5_HIGHT; i ++)
	{
		sen5Neu[i] = (neuron*)xmalloc(hiddenSize * unitNum * sizeof(neuron), "in initNet (sen5neu)");
		N = hiddenSize * unitNum;
		for(j = 0; j < N; j ++)
			sen5Neu[i][j].set();
		//没有padding的卷积操作缩小了每一层大小
		if(unitNum > 1)	unitNum -= conSeq[i] - 1;
	}
	unitNum = 7;
	for(i = 0; i < SEN7_HIGHT; i ++)
	{
		sen7Neu[i] = (neuron*)xmalloc(hiddenSize * unitNum * sizeof(neuron), "in initNet (sen7neu)");
		N = hiddenSize * unitNum;
		for(j = 0; j < N; j ++)
			sen7Neu[i][j].set();
		//没有padding的卷积操作缩小了每一层大小
		if(unitNum > 1)	unitNum -= conSeq[i] - 1;
	}

	// allocate memory and initilization for compression matrix
	compressSyn = (synapse*)xmalloc(2 * hiddenSize * hiddenSize * sizeof(synapse), "in initNet (compress syn)");
	N = 2 * hiddenSize * hiddenSize;
	for(i = 0; i < N; i ++)
		compressSyn[i].weight = random(-0.1, 0.1)+random(-0.1, 0.1)+random(-0.1, 0.1);
	hisNeu = (neuron*)xmalloc(hiddenSize * sizeof(neuron));
	cmbNeu = (neuron*)xmalloc(hiddenSize * 2 * sizeof(neuron));
	// =========== end of data structures for the sentence model ==============

	// allocate memory and initilization for map matrix
	for(i = 0; i < 8; i ++)
	{
		map7Syn[i] = (synapse*)xmalloc(hiddenSize * hiddenSize * sizeof(synapse), "in initNet (map7 syn)");
		N = hiddenSize * hiddenSize;
		for(j = 0; j < N; j ++)
			map7Syn[i][j].weight = random(-0.1, 0.1)+random(-0.1, 0.1)+random(-0.1, 0.1);
	}
	for(i = 0; i < 6; i ++)
	{
		map5Syn[i] = (synapse*)xmalloc(hiddenSize * hiddenSize * sizeof(synapse), "in initNet (map5 syn)");
		N = hiddenSize * hiddenSize;
		for(j = 0; j < N; j ++)
			map5Syn[i][j].weight = random(-0.1, 0.1)+random(-0.1, 0.1)+random(-0.1, 0.1);
	}
	conditionNeu = (neuron*)xmalloc(hiddenSize * sizeof(neuron), "in initNet (condition neu)");

	// allocate memory and initilization for recurrent neural network: input layer, hidden layer and output layer
	inNeu = (neuron*)xmalloc((vocab.getVocabSize() + hiddenSize + hiddenSize) * sizeof(neuron), "in initNet (in neu)");
	N = vocab.getVocabSize() + hiddenSize + hiddenSize;
	for(i = 0; i < N; i ++)
		inNeu[i].set();
	hiddenNeu = (neuron*)xmalloc(hiddenSize * sizeof(neuron), "in initNet (hidden neu)");
	for(i = 0; i < hiddenSize; i ++)
		hiddenNeu[i].set();
	outNeu = (neuron*)xmalloc((vocab.getVocabSize() + classSize)*sizeof(neuron), "in initNet (out neu)");
	N = vocab.getVocabSize() + classSize;
	for(i = 0; i < N; i ++)
		outNeu[i].set();

	hiddenInSyn = (synapse*)xmalloc(hiddenSize * (vocab.getVocabSize() + hiddenSize + hiddenSize) * sizeof(synapse), "in initNet (hidden syn)");
	N = hiddenSize * (vocab.getVocabSize() + hiddenSize + hiddenSize);
	for(i = 0; i < N; i ++)
		hiddenInSyn[i].weight = random(-0.1, 0.1)+random(-0.1, 0.1)+random(-0.1, 0.1);
	outHiddenSyn = (synapse*)xmalloc((vocab.getVocabSize() + classSize) * hiddenSize * sizeof(synapse), "in initNet (hidden syn)");
	N = (vocab.getVocabSize() + classSize) * hiddenSize;
	for(i = 0; i < N; i ++)
		outHiddenSyn[i].weight = random(-0.1, 0.1)+random(-0.1, 0.1)+random(-0.1, 0.1);

	//如果存在已经计算好的词类的文件，就从外部载入，否则自己计算一遍
	if(vocabClassF[0] != 0)
	{
		vocab.loadVocabClass(vocabClassF);
		classStart = (int*)xmalloc(classSize * sizeof(int));
		classEnd = (int*)xmalloc(classSize * sizeof(int));
		vocab.getClassStartEnd(classStart, classEnd, classSize);
	}
	else
		assignClassLabel();

 	bpttHistory = (int*)xmalloc((SEN7_LENGTH + 1) * sizeof(int), "in initNet (bptt history)");
 	bpttHiddenNeu = (neuron*)xmalloc((SEN7_LENGTH + 1) * hiddenSize * sizeof(neuron), "in initNet (bptt hidden neuron)");
 	bpttInHiddenNeu = (neuron*)xmalloc((SEN7_LENGTH + 1) * hiddenSize * sizeof(neuron), "in initNet (bptt input hidden neuron)");
 	bpttHiddenInSyn = (synapse*)xmalloc(hiddenSize * (vocab.getVocabSize() + hiddenSize + hiddenSize) * sizeof(synapse), "in initNet (bptt hidden in sysnapse)");
 	N = hiddenSize * (vocab.getVocabSize() + hiddenSize + hiddenSize);
 	for(i = 0; i < N; i ++)
 		bpttHiddenInSyn[i].weight = 0;
 	bpttConditionNeu = (neuron*)xmalloc((SEN7_LENGTH + 1) * hiddenSize * sizeof(neuron), "in initNet (bptt condition neuron)");

 	senweSyn = (synapse*)xmalloc(vocab.getVocabSize() * hiddenSize * sizeof(synapse), "in initNet (sentence model word embedding sysnapse)");
 	if(mode == TRAIN_MODE)
 	{
		if(randomlyInitSenModelEmbedding)
		{
			N = vocab.getVocabSize() * hiddenSize;
			for(i = 0; i < N; i ++)
				senweSyn[i].weight = random(-0.1, 0.1)+random(-0.1, 0.1)+random(-0.1, 0.1);
		}
		else
		{
			if(voc_arr == NULL) voc_arr = vocab.getVocab();
			int V = vocab.getVocabSize();
			double *embed = new double[hiddenSize];
			for(i = 0; i < V; i ++)
			{
				int size = wdEmbed.getWordEmbedding(voc_arr[i].wd, embed);
				if(size != hiddenSize)
					cout << "word embedding size = " << size << ", hidden size = " << hiddenSize << endl;
				assert(size == hiddenSize);

				for(j = 0; j < hiddenSize; j ++)
					senweSyn[j*V + i].weight = embed[j];
			}
			delete []embed;
		}
 	}

 	initBackup();

 	// for directErr propagate to input layer
	outConditionDSyn = (synapse*)xmalloc((vocab.getVocabSize() + classSize) * hiddenSize * sizeof(synapse), "in initNet (outConditionDSyn)");
	N = (vocab.getVocabSize() + classSize) * hiddenSize;
	for(i = 0; i < N; i ++) outConditionDSyn[i].weight = random(-0.1, 0.1)+random(-0.1, 0.1)+random(-0.1, 0.1);
	bufOutConditionNeu = (neuron*)xmalloc(hiddenSize * (SEN7_LENGTH + 1) * sizeof(neuron), "in initNet (bufOutConditionNeu)");

	// for recurrent context model bptt training
	const int MAX_SEN_POEM = 15;	// this number should be enough: lvshi is 8, quatrain is 4
	conBPTTHis = (neuron*)xmalloc(MAX_SEN_POEM * hiddenSize * sizeof(neuron), "in initNet (con BPTT History)");
	conBPTTCmbHis = (neuron*)xmalloc(MAX_SEN_POEM * hiddenSize * sizeof(neuron), "in initNet (con BPTT combine history)");
	conBPTTCmbSent = (neuron*)xmalloc(MAX_SEN_POEM * hiddenSize * sizeof(neuron), "in initNet (con BPTT combine sentence)");
	bpttHisCmbSyn = (synapse*)xmalloc(hiddenSize * 2 * hiddenSize * sizeof(synapse), "in initNet (bptt History Combine synapse)");
	N = hiddenSize * 2 * hiddenSize;
	for(i = 0; i < N; i ++)
		bpttHisCmbSyn[i].weight = 0;

	if(adaGrad)
		sumGradSquare.init(this);
}

/**
 * @brief
 * 生成字符表
 * @param trainFile
 */
void RNNPG::loadVocab(const char *trainFile)
{
	char buf[1024];
	char word[WDLEN];
	vocab.add2Vocab("</s>");
	FILE *fin = xfopen(trainFile, "r");
	totalPoemCount = 0;
	//将字符流不断读入buf中，取一行，也就是取一句诗
	while( fgets(buf,sizeof(buf),fin) )
	{
		int i = 0;
		//如果没有到达buf的末尾
		while(buf[i] != '\0')
		{
			//如果没有到达buf的末尾并且当前是分隔符，跳过
			while(buf[i] != '\0' && isSep(buf[i])) i ++;
			int j = 0;
			//如果没有到达buf的末尾并且当前不是分隔符
			while(buf[i] != '\0' && !isSep(buf[i]))
			{
				if(j < WDLEN - 1)
					word[j++] = buf[i];
				//这里敢大胆的把i++放在这里，是因为训练文件里i最多只能加一次，所以不会跳过字符导致vocab漏掉了字符
				i ++;
			}
			//向word写入0分隔符，表明word数组已经写满了或者读到了buf的\0
			word[j] = 0;
			if(j > 0)
				//加入词表
				vocab.add2Vocab(word);
		}
		totalPoemCount ++;
		// 添加行尾标识符，every poem has four lines, so 4 end of line!
		vocab.add2Vocab("</s>");
		vocab.add2Vocab("</s>");
		vocab.add2Vocab("</s>");
		vocab.add2Vocab("</s>");
	}
	fclose(fin);
	vocab.reHashVocab(1);
	// vocab.print();
	cout << "load vocabulary done!" << endl;
}

/**
 * @brief
 * 进行CSM前向传播的计算，最后返回一个句子的表达，CSM不考虑句子结尾的</s>
 * @param words 一行诗
 * @param senNeu 指向CSM各层神经元的指针
 * @param SEN_HIGHT CSM层数
 * @return neuron 返回CSM最后一层指向句子的embedding的neuron指针，它指向一个neuron的数组
 */
neuron* RNNPG::sen2vec(const vector<string> &words, neuron **senNeu, int SEN_HIGHT)
{
//	double *embedding = new double[hiddenSize];
	int unitNum = words.size(), i, j;

//	// fill first layer with word embedding...
//	for(i = 0; i < (int)words.size(); i ++)
//	{
//		int weSize = wdEmbed.getWordEmbedding(words[i].c_str(), embedding);
//		if(weSize != hiddenSize)
//		{
//			cout << words[i] << ", weSize = " << weSize << ", hiddenSize = " << hiddenSize << endl;
//			delete []embedding;
//			return NULL;
//		}
//		for(j = 0; j < hiddenSize; j ++)
//			senNeu[0][j*unitNum + i].ac = embedding[j];
//	}
//	delete []embedding;
	// fill first layer with word embedding...
	int V = vocab.getVocabSize();
	//对于每个词
	for(i = 0; i < (int)words.size(); i ++)
	{
		//代表了一个词的id
		int curWord = vocab.getVocabID(words[i].c_str());
		//如果这个词是词汇表里没有的新词就用<R>来代替
		if(curWord == -1) curWord = vocab.getVocabID("<R>");
		//对于每个词对应的神经元的每个维度
		for(j = 0; j < hiddenSize; j ++)
			//使用word embedding初始化CSM输入层的神经元，可以看出
			//内存数据是按照词的顺序存储的
			senNeu[0][j*unitNum + i].ac = senweSyn[V*j + curWord].weight;
	}

	// convolution
	int a, b;
	int unitNumNx = unitNum = words.size();//上一层词的数目
	//对CSM的第i层进行前向传播运算
	//层->层内神经元->词
	for(i = 0; i < SEN_HIGHT - 1; i ++)
	{
		//计算CSM每一层分别有多少个词，词的数目*hiddenSize就是神经元的数目
		unitNumNx = unitNum - (conSeq[i] - 1);//下一层词的数目
//		cout << "unit size = " << unitNumNx << endl;
		//分别对第i层每个词的第a个神经元进行运算
		for(a = 0; a < hiddenSize; a ++)
		{
			int offset = a * unitNum;
			int offsetNx = a * unitNumNx;
			int offsetCon = a * conSeq[i];
			//对第i层对应第a个神经元的每个b词进行运算
			for(b = 0; b < unitNumNx; b ++)
			{
				//对要计算的i+1层神经元进行初始化
				senNeu[i+1][offsetNx + b].ac = 0;
				//对于卷积核的每个维度
				for(j = 0; j < conSeq[i]; j ++)
					senNeu[i+1][offsetNx + b].ac += senNeu[i][offset + b + j].ac * conSyn[i][offsetCon + j].weight;
//				cout << senNeu[i+1][offsetNx + b].ac << ",";
				//神经元使用了sigmod激活函数
				senNeu[i+1][offsetNx + b].fun_ac();
//				cout << senNeu[i+1][offsetNx + b].ac << " ";
			}
//			cout << endl;
		}
//		cout << endl;
		unitNum = unitNumNx;
	}
	//最后一层应该只有一个词
	assert(unitNumNx == 1);
	//返回最后一层句子的embedding
	return senNeu[SEN_HIGHT-1];
}

/**
 * @brief
 * 初始化CSM里的各个神经元，将其初值ac和er设置为0
 * @param senLen 诗里每一句的长度
 */
void RNNPG::initSent(int senLen)
{
	int unitNum = senLen, i, j, N;
	int SEN_HIGHT = senLen == 5 ? SEN5_HIGHT : SEN7_HIGHT;
	neuron** senNeu = senLen == 5 ? sen5Neu : sen7Neu;
	for(i = 0; i < SEN_HIGHT; i ++)
	{
		N = hiddenSize * unitNum;
		for(j = 0; j < N; j ++)
			senNeu[i][j].set();
		if(unitNum > 1)	unitNum -= conSeq[i] - 1;
	}
//	clearNeurons(hisNeu, hiddenSize, 3);
//	clearNeurons(cmbNeu, hiddenSize * 2, 3);
//	clearNeurons(conditionNeu, hiddenSize, 3);
//	int V = vocab.getVocabSize();
//	clearNeurons(inNeu, V + hiddenSize + hiddenSize, 3);
//	clearNeurons(hiddenNeu, hiddenSize, 3);
//	clearNeurons(outNeu, V + classSize, 3);
//
//	memset(bpttHistory, 0xff, sizeof(int)*(SEN7_LENGTH+1));
//	// clearNeurons(bpttHiddenNeu, hiddenSize * (SEN7_LENGTH+1), 3);
}

/**
 * @brief
 * 主要是重设RGM
 * 重设输入层inNeu里的来自RCM的u_i^j和来自RGM的r_{j-1}的ac和er，将其ac设置成stableAC
 * 清空hiddenNeu(RGM隐含层r_j)的ac和er
 * 清空bpttHistory，每个位置存上-1，相当于没有见过的词
 */
void RNNPG::flushNet()
{
	////////////////////////////////////////////////////////////////////////////////////
	int V = vocab.getVocabSize();
	// clearNeurons(inNeu, V + hiddenSize + hiddenSize, 3);
	clearNeurons(inNeu + V, hiddenSize, 3);
	int layer1size = V + hiddenSize + hiddenSize;
	for(int i = V + hiddenSize; i < layer1size; i ++)
	{
		inNeu[i].ac = stableAC;			// like in rnnlm, last hidden layer is initialized to vector of 0.1 values to prevent unstability
		inNeu[i].er = 0;
	}
	clearNeurons(hiddenNeu, hiddenSize, 3);
	// clearNeurons(outNeu, V + classSize, 3);

	//相当于给bpttHistory每个位置存的值都是-1，装逼！
	memset(bpttHistory, 0xff, sizeof(int)*(SEN7_LENGTH+1));
	// clearNeurons(bpttHiddenNeu, hiddenSize * (SEN7_LENGTH+1), 3);
}

/**
 * @brief
 * 计算每个词所属的类别
 */
void RNNPG::assignClassLabel()
{
	classStart = (int*)xmalloc(classSize * sizeof(int));
	classEnd = (int*)xmalloc(classSize * sizeof(int));
	memset(classStart, 0, classSize * sizeof(int));
	memset(classEnd, 0, classSize * sizeof(int));
	voc_arr = vocab.getVocab();
	int V = vocab.getVocabSize(), i;
	int tot_freq = 0;
	for(i = 0; i < V; i ++) tot_freq += voc_arr[i].freq;
	double prob = 0; int classIndex = 0;
	classStart[0] = 0;
	for(i = 0; i < V; i ++)
	{
		//prob某个词出现的概率，voc_arr里的词是按照出现频率由低到高进行排序的
		prob += voc_arr[i].freq / (double)tot_freq;
		if(prob > 1) prob = 1;
		voc_arr[i].classIndex = classIndex;
		if(prob > (double)(classIndex + 1)/classSize)
			if(classIndex < classSize - 1)
			{
				classEnd[classIndex] = i + 1;
				classIndex ++;
				classStart[classIndex] = i + 1;
			}
	}
	classEnd[classIndex] = V;
	// just for test
//	for(i = 0; i < classSize; i ++)
//		cout << "class " << i << ", " << classStart[i] << ", " << classEnd[i] << endl;

	// print information
	// vocab.print();
}

/**
 * @brief
 * 根据已有的诗歌中的句子来计算RGM，属于训练过程，通过directError来控制是否使用RCM
 * @param lastWord 上一个词在词汇表中的ID
 * @param curWord 当前正在生成的词在词汇表中的ID
 * @param wordPos 当前正在处理一句诗里的第几个词
 * @param mapSyn RCM中的U_j矩阵
 */
void RNNPG::computeNet(int lastWord, int curWord, int wordPos, synapse **mapSyn)
{
	clearNeurons(conditionNeu, hiddenSize, 1);
	matrixXvector(conditionNeu, hisNeu, mapSyn[wordPos], hiddenSize, 0, hiddenSize, 0, hiddenSize, 0);
	funACNeurons(conditionNeu, hiddenSize);
	int V = vocab.getVocabSize();
	memcpy(inNeu + V, conditionNeu, hiddenSize * sizeof(neuron));
	// go back later...

	// input layer to hidden layer，输入层到隐含层
	clearNeurons(hiddenNeu, hiddenSize, 1);
	matrixXvector(hiddenNeu, inNeu, hiddenInSyn, V + hiddenSize + hiddenSize, 0, hiddenSize, V, V + hiddenSize + hiddenSize, 0);
	int i, N = V + hiddenSize + hiddenSize;
	for(i = 0; i < hiddenSize; i ++)
	{
		//计算r_j中X\cdot e(w_j)，加到还没有经过激活的r_j中
		hiddenNeu[i].ac += hiddenInSyn[i*N + lastWord].weight;
		hiddenNeu[i].fun_ac();
	}

	// hidden layer to output layer
	// 1. hidden layer to class
	clearNeurons(outNeu + V, classSize, 1);            // clearNeurons(outNeu, classSize, 1); I used outNeu by mistake (it should be outNeu + V)
	matrixXvector(outNeu, hiddenNeu, outHiddenSyn, hiddenSize, V, V + classSize, 0, hiddenSize, 0);

	if(directError)
		//使用RCM直接去预测词类
		// 1. condition layer (in the input layer) to the output layer -- for classes
		matrixXvector(outNeu, inNeu + V, outConditionDSyn, hiddenSize, V, V + classSize, 0, hiddenSize, 0);

	// compute softmax probability
	//计算P(word_class|context)
	double sum = 0;
	for(i = 0; i < classSize; i ++)
	{
		//控制exp的值不能溢出，将自变量限制在[-50,50]之间
		if(outNeu[V+i].ac < -50) outNeu[V+i].ac = -50;
		if(outNeu[V+i].ac > 50) outNeu[V+i].ac = 50;
		outNeu[V+i].ac = FAST_EXP(outNeu[V+i].ac);
		sum += outNeu[V + i].ac;
	}
	for(i = 0; i < classSize; i ++)
		outNeu[V + i].ac /= sum;

	// 2. hidden layer to words
	int curClassIndex = voc_arr[curWord].classIndex;//计算这句诗的wordPos的词所属的类的标签
	clearNeurons(outNeu + classStart[curClassIndex], classEnd[curClassIndex] - classStart[curClassIndex], 1);
	matrixXvector(outNeu, hiddenNeu, outHiddenSyn, hiddenSize, classStart[curClassIndex], classEnd[curClassIndex], 0, hiddenSize, 0);

	if(directError)
		//使用RCM预测词类中的词
		// 1. condition layer (in the input layer) to the output layer -- for words
		matrixXvector(outNeu, inNeu + V, outConditionDSyn, hiddenSize, classStart[curClassIndex], classEnd[curClassIndex], 0, hiddenSize, 0);

	//计算P(word|word_class,context)，只对curWord所属的word_class里所有的词计算softmax，并且这样分割意味着不同的word_class包含的词是没有交集的
	sum = 0;
	for(i = classStart[curClassIndex]; i < classEnd[curClassIndex]; i ++)
	{
		//控制exp的值不能溢出，将自变量限制在[-50,50]之间
		if(outNeu[i].ac < -50) outNeu[i].ac = -50;
		if(outNeu[i].ac > 50) outNeu[i].ac = 50;
		outNeu[i].ac = FAST_EXP(outNeu[i].ac);
		sum += outNeu[i].ac;
	}
	for(i = classStart[curClassIndex]; i < classEnd[curClassIndex]; i ++)
		outNeu[i].ac /= sum;
}

/**
 * @brief
 * 将误差传递到RCM和CSM中，这里并没有使用BPTT来训练RCM
 * @param senLen 句子的长度
 */
void RNNPG::learnSent(int senLen)
{
	//反向传播RCM
	double beta2 = alpha * beta;
	int i, j, N = hiddenSize + hiddenSize;
	//将误差传递到$M\begin{bmatrix}v_i\\h_{i-1}\end{bmatrix}$上
	for(i = 0; i < hiddenSize; i ++)
		hisNeu[i].er *= hisNeu[i].ac * (1 - hisNeu[i].ac);
	//清空v_i的error
	clearNeurons(cmbNeu + hiddenSize, hiddenSize, 2);

	// back propagate error from the history representation to sentence top layer (the final representation of the sentence)，反向传播到v_i中去
	matrixXvector(cmbNeu, hisNeu, compressSyn, hiddenSize + hiddenSize, 0, hiddenSize, hiddenSize, hiddenSize + hiddenSize, 1);
	// update compress matrix
//	if(wordCounter % 10 == 0)
//	{
		//更新M矩阵
		for(i = 0; i < hiddenSize; i ++)
			for(j = 0; j < N; j ++)
				compressSyn[i * N + j].weight += alpha * hisNeu[i].er * cmbNeu[j].ac - compressSyn[i * N + j].weight * beta2;
//	}
//	else
//	{
//		for(i = 0; i < hiddenSize; i ++)
//			for(j = 0; j < N; j ++)
//				compressSyn[i * N + j].weight += alpha * hisNeu[i].er * cmbNeu[j].ac;
//	}

	// 反向传播CSM
	// error propagate in sentence model
	neuron **senNeu = senLen == 5 ? sen5Neu : sen7Neu;
	int SEN_HIGHT = senLen == 5 ? SEN5_HIGHT : SEN7_HIGHT;
	//unitNumNx上一层卷积后的单元数目，unitNum是下一层卷积后的单元数目
	int unitNum = 1, unitNumNx = 1, a = -1, b = -1;
	// 将误差传递到CSM的顶层里面
	for(i = 0; i < hiddenSize; i ++)
		senNeu[SEN_HIGHT - 1][i].er = cmbNeu[hiddenSize + i].er;
	for(i = SEN_HIGHT - 2; i >= 0; i --)
	{
		unitNumNx = unitNum + (conSeq[i] - 1);
		int offset = 0, offsetNx = 0, offsetCon = 0;
		// for readability, I just compute the deviation seperately
		// 经过激活函数，将误差传递到$\sum_{i=1}^nT^l_{:,j+i-1} \odot C^{l,n}_{:,i}$上
		for(a = 0; a < hiddenSize; a ++)
			for(b = 0; b < unitNum; b ++)
			{
				offset = a * unitNum;
				senNeu[i + 1][offset + b].er *= senNeu[i + 1][offset + b].ac * (1 - senNeu[i + 1][offset + b].ac);
			}

		// compute the back propagate error
		// 如果！（到第一层并且不在更新CSM的时候更新Word embedding矩阵）
		if(i != 0 || !fixSentenceModelFirstLayer)
		{
			//将误差传递到第i层卷积层
			for(a = 0; a < hiddenSize; a ++)
			{
				offset = a * unitNum;
				offsetNx = a * unitNumNx;
				offsetCon = a * conSeq[i];
				for(b = 0; b < unitNum; b ++)
				{
					for(j = 0; j < conSeq[i]; j ++)
						senNeu[i][offsetNx + b + j].er += senNeu[i + 1][offset + b].er * conSyn[i][offsetCon + j].weight;
				}
			}
			//限制误差的范围，防止梯度爆炸
			for(a = 0; a < hiddenSize; a ++)
			{
				offsetNx = a * unitNumNx;
				for(b = 0; b < unitNumNx; b ++)
				{
					//assert(senNeu[i][offsetNx + b].er >= -15 && senNeu[i][offsetNx + b].er <= 15);
					if(senNeu[i][offsetNx + b].er < -15) senNeu[i][offsetNx + b].er = -15;
					if(senNeu[i][offsetNx + b].er > 15) senNeu[i][offsetNx + b].er = 15;
				}
			}
		}

		// 更新卷积核
		// update the matrix, at this point we do NOT consider the L2 normalization term
		for(a = 0; a < hiddenSize; a ++)
		{
			offset = a * unitNum;
			offsetNx = a * unitNumNx;
			offsetCon = a * conSeq[i];
			for(j = 0; j < conSeq[i]; j ++)
				conSynOffset[i][offsetCon + j].weight = 0;
			for(b = 0; b < unitNum; b ++)
			{
				for(j = 0; j < conSeq[i]; j ++)
					conSynOffset[i][offsetCon + j].weight += alpha * senNeu[i + 1][offset + b].er * senNeu[i][offsetNx + b + j].ac;
			}
		}
		// update the matrix with L2 normalization term
		for(a = 0; a < hiddenSize; a ++)
		{
			offsetCon = a * conSeq[i];
			for(j = 0; j < conSeq[i]; j ++)
				conSyn[i][offsetCon + j].weight += conSynOffset[i][offsetCon + j].weight - beta2 * conSyn[i][offsetCon + j].weight;
		}

		unitNum = unitNumNx;
	}
	// cout << unitNumNx << endl;
	// 第一层的大小应该和句子的长度是一样的
	assert(unitNumNx == senLen);

	//如果到了第一层并且不在更新CSM的时候更新Word embedding矩阵，就直接返回
	if(fixSentenceModelFirstLayer)
		return;
	int V = vocab.getVocabSize();
	//newWordCounter不考虑结尾的</s>
	int newWordCounter = wordCounter - senLen - 1;
	//更新word embedding矩阵
	for(i = 0; i < unitNumNx; i ++)
	{
		//word是第i个位置的词的id
		int word = bpttHistory[i + 1];	// because bpttHistory recorded lastWord, not curWord
		newWordCounter ++;
		//每10个词正则化一次
		if(newWordCounter % 10 == 0)
		{
			for(j = 0; j < hiddenSize; j ++)
			{
				senNeu[0][j*unitNumNx + i].er *= senNeu[0][j*unitNumNx + i].ac * (1 - senNeu[0][j*unitNumNx + i].ac);
				senweSyn[V*j + word].weight += alpha * senNeu[0][j*unitNumNx + i].er - beta2 * senweSyn[V*j + word].weight;
			}
		}
		else
		{
			for(j = 0; j < hiddenSize; j ++)
			{
				senNeu[0][j*unitNumNx + i].er *= senNeu[0][j*unitNumNx + i].ac * (1 - senNeu[0][j*unitNumNx + i].ac);
				senweSyn[V*j + word].weight += alpha * senNeu[0][j*unitNumNx + i].er;
			}
		}
	}
}

/**
 * @brief
 * 读取完了一整句诗之后，使用BPTT来训练RCM
 * @param senLen 句子的长度
 */
void RNNPG::learnSentBPTT(int senLen)
{
	/*
	 neuron* conBPTTHis;
	neuron* conBPTTCmbHis;
	neuron* conBPTTCmbSent;
	 */
	//将h_i拷贝进conBPTTHis + hiddenSize * contextBPTTSentNum的位置
	copyNeurons(conBPTTHis + hiddenSize * contextBPTTSentNum, hisNeu, hiddenSize, 3);
	//将h_{i-1}拷贝进conBPTTCmbHis + hiddenSize * contextBPTTSentNum的位置
	copyNeurons(conBPTTCmbHis + hiddenSize * contextBPTTSentNum, cmbNeu, hiddenSize, 3);
	//将v_i拷贝进conBPTTCmbSent + hiddenSize * contextBPTTSentNum的位置
	copyNeurons(conBPTTCmbSent + hiddenSize * contextBPTTSentNum, cmbNeu + hiddenSize, hiddenSize, 3);

	//如果还没到一首诗的最后一句话就返回
	if(!isLastSentOfPoem) return;

	//到了最后一句诗，开始处理
	double beta2 = alpha * beta;
	int i, j, N = hiddenSize + hiddenSize;
	for(int step = contextBPTTSentNum; step > 0; step --)
	{
		//将误差传递到$M \cdot \begin{bmatrix}v_i\\h_{i-1}\end{bmatrix}$上去
		for(i = 0; i < hiddenSize; i ++)
			hisNeu[i].er *= hisNeu[i].ac * (1 - hisNeu[i].ac);
		//清空v_i的误差
		clearNeurons(cmbNeu + hiddenSize, hiddenSize, 2);
		//将误差传递到v_i上，back propagate error from the history representation to sentence top layer (the final representation of the sentence)
		matrixXvector(cmbNeu, hisNeu, compressSyn, hiddenSize + hiddenSize, 0, hiddenSize, hiddenSize, hiddenSize + hiddenSize, 1);
		// update compress matrix
	//	if(wordCounter % 10 == 0)
	//	{
			//将BPTT过程中对M累积的误差存在bpttHisCmbSyn中
			for(i = 0; i < hiddenSize; i ++)
				for(j = 0; j < N; j ++)
				{
					// compressSyn[i * N + j].weight += alpha * hisNeu[i].er * cmbNeu[j].ac - compressSyn[i * N + j].weight * beta2;
					// now it is bptt, so compressSyn matrix will not be updated, util all the bptt step is done. The gradient will be store
					bpttHisCmbSyn[i * N + j].weight += alpha * hisNeu[i].er * cmbNeu[j].ac;
				}
	//	}
	//	else
	//	{
	//		for(i = 0; i < hiddenSize; i ++)
	//			for(j = 0; j < N; j ++)
	//				compressSyn[i * N + j].weight += alpha * hisNeu[i].er * cmbNeu[j].ac;
	//	}

		// error propagate in sentence model,以下和learnSent中将误差传递至CSM中的过程差不多
		//------learnSent(Start)------
		neuron **senNeu = senLen == 5 ? sen5Neu : sen7Neu;
		int SEN_HIGHT = senLen == 5 ? SEN5_HIGHT : SEN7_HIGHT;
		int unitNum = 1, unitNumNx = 1, a = -1, b = -1;
		for(i = 0; i < hiddenSize; i ++)
			senNeu[SEN_HIGHT - 1][i].er = cmbNeu[hiddenSize + i].er;
		for(i = SEN_HIGHT - 2; i >= 0; i --)
		{
			unitNumNx = unitNum + (conSeq[i] - 1);
			int offset = 0, offsetNx = 0, offsetCon = 0;
			// for readability, I just compute the deviation seperately
			for(a = 0; a < hiddenSize; a ++) for(b = 0; b < unitNum; b ++)
			{
				offset = a * unitNum;
				senNeu[i + 1][offset + b].er *= senNeu[i + 1][offset + b].ac * (1 - senNeu[i + 1][offset + b].ac);
			}

			// compute the back propagate error
			if(i != 0 || !fixSentenceModelFirstLayer)
			{
				for(a = 0; a < hiddenSize; a ++)
				{
					offset = a * unitNum;
					offsetNx = a * unitNumNx;
					offsetCon = a * conSeq[i];
					for(b = 0; b < unitNum; b ++)
					{
						for(j = 0; j < conSeq[i]; j ++)
							senNeu[i][offsetNx + b + j].er += senNeu[i + 1][offset + b].er * conSyn[i][offsetCon + j].weight;
					}
				}
				for(a = 0; a < hiddenSize; a ++)
				{
					offsetNx = a * unitNumNx;
					for(b = 0; b < unitNumNx; b ++)
					{
						//assert(senNeu[i][offsetNx + b].er >= -15 && senNeu[i][offsetNx + b].er <= 15);
						if(senNeu[i][offsetNx + b].er < -15) senNeu[i][offsetNx + b].er = -15;
						if(senNeu[i][offsetNx + b].er > 15) senNeu[i][offsetNx + b].er = 15;
					}
				}
			}

			// update the matrix, at this point we do NOT consider the L2 normalization term
			for(a = 0; a < hiddenSize; a ++)
			{
				offset = a * unitNum;
				offsetNx = a * unitNumNx;
				offsetCon = a * conSeq[i];
				for(j = 0; j < conSeq[i]; j ++)
					conSynOffset[i][offsetCon + j].weight = 0;
				for(b = 0; b < unitNum; b ++)
				{
					for(j = 0; j < conSeq[i]; j ++)
						conSynOffset[i][offsetCon + j].weight += alpha * senNeu[i + 1][offset + b].er * senNeu[i][offsetNx + b + j].ac;
				}
			}
			// update the matrix with L2 normalization term
			for(a = 0; a < hiddenSize; a ++)
			{
				offsetCon = a * conSeq[i];
				for(j = 0; j < conSeq[i]; j ++)
					conSyn[i][offsetCon + j].weight += conSynOffset[i][offsetCon + j].weight - beta2 * conSyn[i][offsetCon + j].weight;
			}

			unitNum = unitNumNx;
		}
		// cout << unitNumNx << endl;
		assert(unitNumNx == senLen);

		if(fixSentenceModelFirstLayer)
			return;
		int V = vocab.getVocabSize();
		int newWordCounter = wordCounter - senLen - 1;
		for(i = 0; i < unitNumNx; i ++)
		{
			int word = bpttHistory[i + 1];	// because bpttHistory recorded lastWord, not curWord
			newWordCounter ++;
			if(newWordCounter % 10 == 0)
			{
				for(j = 0; j < hiddenSize; j ++)
				{
					senNeu[0][j*unitNumNx + i].er *= senNeu[0][j*unitNumNx + i].ac * (1 - senNeu[0][j*unitNumNx + i].ac);
					senweSyn[V*j + word].weight += alpha * senNeu[0][j*unitNumNx + i].er - beta2 * senweSyn[V*j + word].weight;
				}
			}
			else
			{
				for(j = 0; j < hiddenSize; j ++)
				{
					senNeu[0][j*unitNumNx + i].er *= senNeu[0][j*unitNumNx + i].ac * (1 - senNeu[0][j*unitNumNx + i].ac);
					senweSyn[V*j + word].weight += alpha * senNeu[0][j*unitNumNx + i].er;
				}
			}
		}
		//------learnSent(End)-------
		//从这里开始是和learnSent不同的地方

		// now this is the time for bptt -- hisNeu
		// 清空h_{i-1}的误差
		clearNeurons(cmbNeu, hiddenSize, 2);
		// back propagate error from the history representation to sentence top layer (the final representation of the sentence)
		// 将误差传递到cmbNeu中的h_{i-1}中去
		matrixXvector(cmbNeu, hisNeu, compressSyn, hiddenSize + hiddenSize, 0, hiddenSize, 0, hiddenSize, 1);
		// update compress matrix, already done at the beginning
		if(step > 1)
		{
			for(i = 0; i < hiddenSize; i ++)
			{
				hisNeu[i].er = cmbNeu[i].er + conBPTTHis[(step - 1) * hiddenSize + i].er;//将误差沿着时间传递
				hisNeu[i].ac = conBPTTHis[(step - 1) * hiddenSize + i].ac;
				cmbNeu[i].ac = conBPTTCmbHis[(step - 1) * hiddenSize + i].ac;
				cmbNeu[hiddenSize + i].ac = conBPTTCmbSent[(step - 1) * hiddenSize + i].ac;
			}
		}
	}

	// update compressSyn matrix: this is a large step
	N = hiddenSize + hiddenSize;
	for(i = 0; i < hiddenSize; i ++)
		for(j = 0; j < N; j ++)
		{
			compressSyn[i * N + j].weight += bpttHisCmbSyn[i * N + j].weight - compressSyn[i * N + j].weight * beta2;
			bpttHisCmbSyn[i * N + j].weight = 0;
		}
}

/**
 * @brief
 * 学习整个网络的过程
 * 如果还没有到一句诗的结尾，就什么都不干，说明Y是对每个字都更新，而其他的参数是对每句诗做更新
 * @param lastWord 上一个词对应于词表中的ID
 * @param curWord 当前词对应于词表中的ID
 * @param wordPos 正在处理的一个词在一句诗里的位置
 * @param senLen 诗句的长度，不包含结尾的定界符"</s>"
 */
void RNNPG::learnNet(int lastWord, int curWord, int wordPos, int senLen)
{
	double beta2 = alpha * beta;
	int curClassIndex = voc_arr[curWord].classIndex, i = 0, j = 0, V = vocab.getVocabSize(), N = 0, offset = 0;
	//误差传递到softmax激活之前的线性单元$-\delta ^{(softmax)}=y^{(label)}-a^{(softmax)}$
	// error at output layer. 1. error on words
	for(i = classStart[curClassIndex]; i < classEnd[curClassIndex]; i ++)
		outNeu[i].er = 0 - outNeu[i].ac;
	outNeu[curWord].er = 1 - outNeu[curWord].ac;
	// error at output layer. 2. error on classes
	N = V + classSize;
	for(i = V; i < N; i ++)
		outNeu[i].er = 0 - outNeu[i].ac;
	outNeu[V + curClassIndex].er = 1 - outNeu[V + curClassIndex].ac;

	clearNeurons(hiddenNeu, hiddenSize, 2);

	// error backpropagation to hidden layer,对应于$-Y^T\delta ^{(softmax)}$
	// 1. error from words to hidden
	matrixXvector(hiddenNeu, outNeu, outHiddenSyn, hiddenSize, classStart[curClassIndex], classEnd[curClassIndex], 0, hiddenSize, 1);
	// 2. error from classes to hidden
	matrixXvector(hiddenNeu, outNeu, outHiddenSyn, hiddenSize, V, V + classSize, 0, hiddenSize, 1);

	if(directError)
	{
		//使用RCM生成，此处将误差直接传导到u_i^j上
		// bufOutConditionNeu
		clearNeurons(bufOutConditionNeu + (wordPos * hiddenSize), hiddenSize, 2);
		// error back propagate to conditionNeu
		// 1. error from words to conditionNeu
		matrixXvector(bufOutConditionNeu + (wordPos * hiddenSize), outNeu, outConditionDSyn, hiddenSize, classStart[curClassIndex], classEnd[curClassIndex], 0, hiddenSize, 1);
		// 2. error from classes to conditionNeu
		matrixXvector(bufOutConditionNeu + (wordPos * hiddenSize), outNeu, outConditionDSyn, hiddenSize, V, V + classSize, 0, hiddenSize, 1);
	}

	// updating the matrix outHiddenSyn, since we already have the error at output layer and the activation in the hidden layer
	// we update the weight per word rather than per sentence for faster increase in likelihood. Perhaps it will be modified to per sentence update later
	// $Y=Y-\alpha\bigtriangledown_YL=Y-\alpha\cdot\delta ^{(softmax)}r_j^T$
	// update submatrix of words to hidden layer,更新对应于相应类别的词的Y矩阵
	offset = classStart[curClassIndex] * hiddenSize;
	for(i = classStart[curClassIndex]; i < classEnd[curClassIndex]; i ++)
	{
		if(wordCounter % 10 == 0)
			//每学习10个字正则化一次
			for(j = 0; j < hiddenSize; j ++) outHiddenSyn[offset + j].weight += alpha * outNeu[i].er * hiddenNeu[j].ac - beta2*outHiddenSyn[offset + j].weight;
		else
			for(j = 0; j < hiddenSize; j ++) outHiddenSyn[offset + j].weight += alpha * outNeu[i].er * hiddenNeu[j].ac;
		offset += hiddenSize;
	}
	// update submatrix of classes to hidden layer，更新相应类别的Y矩阵
	N = V + classSize;
	offset = V * hiddenSize;
	for(i = V; i < N; i ++)
	{
		if(wordCounter % 10 == 0)
			//每学习10个字正则化一次
			for(j = 0; j < hiddenSize; j ++) outHiddenSyn[offset + j].weight += alpha * outNeu[i].er * hiddenNeu[j].ac - beta2*outHiddenSyn[offset + j].weight;
		else
			for(j = 0; j < hiddenSize; j ++) outHiddenSyn[offset + j].weight += alpha * outNeu[i].er * hiddenNeu[j].ac;
		offset += hiddenSize;
	}

	if(directError)
	{
		// update the matrix outConditionDSyn
		// update submatrix of words to condition layer
		offset = classStart[curClassIndex] * hiddenSize;
		for(i = classStart[curClassIndex]; i < classEnd[curClassIndex]; i ++)
		{
			// how can I make such stupid bug !!! hiddenNeu[V + j].ac is obviously incorrect !!! inNeu[V + j].ac
			if(wordCounter % 10 == 0)
				for(j = 0; j < hiddenSize; j ++) outConditionDSyn[offset + j].weight += alpha * outNeu[i].er * inNeu[V + j].ac - beta2*outConditionDSyn[offset + j].weight;
			else
				for(j = 0; j < hiddenSize; j ++) outConditionDSyn[offset + j].weight += alpha * outNeu[i].er * inNeu[V + j].ac;
			offset += hiddenSize;
		}

		// update submatrix of classes to condition layer
		N = V + classSize;
		offset = V * hiddenSize;
		for(i = V; i < N; i ++)
		{
			// how can I make such stupid bug !!! hiddenNeu[V + j].ac is obviously incorrect !!! inNeu[V + j].ac
			if(wordCounter % 10 == 0)
				for(j = 0; j < hiddenSize; j ++) outConditionDSyn[offset + j].weight += alpha * outNeu[i].er * inNeu[V + j].ac - beta2*outConditionDSyn[offset + j].weight;
			else
				for(j = 0; j < hiddenSize; j ++) outConditionDSyn[offset + j].weight += alpha * outNeu[i].er * inNeu[V + j].ac;
			offset += hiddenSize;
		}
	}

	// this is for back propagation through time，开始BPTT过程了
	bpttHistory[wordPos] = lastWord;	// store the last word,在wordPos处存储上一个词的ID
	memcpy(bpttHiddenNeu + (wordPos * hiddenSize), hiddenNeu, sizeof(neuron)*hiddenSize);	// store the hidden layer，将r_j放进了bpttHiddenNeu + (wordPos * hiddenSize)
	memcpy(bpttInHiddenNeu + (wordPos * hiddenSize), inNeu + (V + hiddenSize), sizeof(neuron)*hiddenSize);	// store the hidden units in input layer (previous hidden layer),将r_{j-1}放进了bpttInHiddenNeu + (wordPos * hiddenSize)
	memcpy(bpttConditionNeu + (wordPos * hiddenSize), inNeu + V, sizeof(neuron)*hiddenSize);	// store the condition units in input layer，将u_i^j放进了bpttConditionNeu + (wordPos * hiddenSize)
	// 如果还没有到一句诗的结尾，就什么都不干，说明Y是对每个字都更新，而其他的参数是对每句诗做更新
	if(curWord != 0)
		return;
	// if this is the end of sentence, then let's do it，此时wordPos=诗句的长度+1
	int lword = -1, layer1Size = V + hiddenSize + hiddenSize;//lword表示正在处理的词的上一个词的ID
	synapse **mapSyn = NULL;
	mapSyn = senLen == 5 ? map5Syn : map7Syn;
	for(int step = wordPos; step >= 0; step --)
	{
		// take care of vocabulary and recurrent part in input layer first
		// bpttHiddenInSyn[]
		for(i = 0; i < hiddenSize; i ++)
			//sigmoid的导数
			hiddenNeu[i].er *= hiddenNeu[i].ac * (1 - hiddenNeu[i].ac);
		lword = bpttHistory[step];
		// X->accumulate deviations for input matrix, X，只更新lword
		for(i = 0; i < hiddenSize; i ++)
			bpttHiddenInSyn[i * layer1Size + lword].weight += alpha * hiddenNeu[i].er;

		// 更新r_{j-1}
		clearNeurons(inNeu + (V+hiddenSize), hiddenSize, 2);
		matrixXvector(inNeu, hiddenNeu, hiddenInSyn, layer1Size, 0, hiddenSize, V + hiddenSize, layer1Size, 1);

		// R->accumulate deviations for hidden matrix, R
		for(i = 0; i < hiddenSize; i ++)
			for(j = V + hiddenSize; j < layer1Size; j ++)
				bpttHiddenInSyn[i*layer1Size + j].weight += alpha * hiddenNeu[i].er * inNeu[j].ac;

		// now we take care the condition part in the input layer
		// 更新u_i^j，back propagate the error to condition part
		clearNeurons(inNeu + V, hiddenSize, 2);
		matrixXvector(inNeu, hiddenNeu, hiddenInSyn, layer1Size, 0, hiddenSize, V, V + hiddenSize, 1);

		// H->accumulate deviations for condition matrix, H
		N = V + hiddenSize;
		for(i = 0; i < hiddenSize; i ++)
			for(j = V; j < N; j ++)
				bpttHiddenInSyn[i*layer1Size + j].weight += alpha * hiddenNeu[i].er * inNeu[j].ac;

		// 只使用RCM
		if(directError)
		{
			for(i = 0; i < hiddenSize; i ++)
				inNeu[V + i].er += bufOutConditionNeu[step * hiddenSize + i].er;
		}

		// 计算u_i^j的激活函数，将误差传导到U_j \cdot h_i上
		for(i = V; i < N; i ++)
			inNeu[i].er *= inNeu[i].ac * (1 - inNeu[i].ac);

		if(perSentUpdate)
			clearNeurons(hisNeu, hiddenSize, 2);

		// 注意之前只有在perSentUpdate时才清空，如果不是perSentUpdate，这里误差将会累积下去，等这一句话接受之后一起向前传递，watch that the error in hisNeu must be inilizated to zero at the beginning of dealing with each sentence
		matrixXvector(hisNeu, inNeu + V, mapSyn[step], hiddenSize, 0, hiddenSize, 0, hiddenSize, 1);

		// acumulate deviations for map matrix
		// 每10个字做一次正则化
		// 训练U_j矩阵
		if(wordCounter % 10 == 0)
		{
			for(i = 0; i < hiddenSize; i ++)
				for(j = 0; j < hiddenSize; j ++)
					mapSyn[step][i * hiddenSize + j].weight += alpha * inNeu[V+i].er * hisNeu[j].ac - mapSyn[step][i * hiddenSize + j].weight * beta2;
		}
		else
		{
			for(i = 0; i < hiddenSize; i ++)
				for(j = 0; j < hiddenSize; j ++)
					mapSyn[step][i * hiddenSize + j].weight += alpha * inNeu[V+i].er * hisNeu[j].ac;
		}

		if(perSentUpdate)
			// 训练RCM,和CSM
			learnSent(senLen);

		// 当已经到了第一个词的时候，做上面的步骤，不做下面的步骤，避免越界
		if(step == 0) continue;
		// propagate error to previous layer
		for(i = 0; i < hiddenSize; i ++)
		{
			hiddenNeu[i].er = inNeu[V + hiddenSize + i].er + bpttHiddenNeu[(step - 1)*hiddenSize + i].er;
			hiddenNeu[i].ac = bpttHiddenNeu[(step - 1)*hiddenSize + i].ac;
		// restore the recurrent part in input layer
			inNeu[V + hiddenSize + i].ac = bpttInHiddenNeu[(step - 1)*hiddenSize + i].ac;
		// restore the condition part in input layer
			inNeu[V + i].ac = bpttConditionNeu[(step - 1)*hiddenSize + i].ac;
		}
	}

	// restore hidden layer
	memcpy(hiddenNeu, bpttHiddenNeu + (wordPos * hiddenSize), sizeof(neuron)*hiddenSize);


	// update input matrix, X, condition Matrix, H and recurrent matrix, R
	for(i = 0; i < hiddenSize; i ++)
	{
//		if(wordCounter % 10 == 0)
//		{
			//用于更新Word Embedding矩阵X
			for(j = 0; j <= wordPos; j ++)
			{
				lword = bpttHistory[j];
				hiddenInSyn[i*layer1Size + lword].weight += bpttHiddenInSyn[i*layer1Size + lword].weight - hiddenInSyn[i*layer1Size + lword].weight * beta2;
				bpttHiddenInSyn[i*layer1Size + lword].weight = 0;
			}
//		}
//		else
//		{
//			for(j = 0; j <= wordPos; j ++)
//			{
//				lword = bpttHistory[j];
//				hiddenInSyn[i*layer1Size + lword].weight += bpttHiddenInSyn[i * layer1Size + lword].weight;
//				bpttHiddenInSyn[i * layer1Size + lword].weight = 0;
//			}
//		}

//		if(wordCounter % 10 == 0)
//		{
			//用于更新H矩阵
			N = V + hiddenSize;
			for(j = V; j < N; j ++)
			{
				hiddenInSyn[i*layer1Size + j].weight += bpttHiddenInSyn[i*layer1Size + j].weight - hiddenInSyn[i*layer1Size + j].weight * beta2;
				bpttHiddenInSyn[i*layer1Size + j].weight = 0;
			}
//		}
//		else
//		{
//			N = V + hiddenSize;
//			for(j = V; j < N; j ++)
//			{
//				hiddenInSyn[i*layer1Size + j].weight += bpttHiddenInSyn[i*layer1Size + j].weight;
//				bpttHiddenInSyn[i*layer1Size + j].weight = 0;
//			}
//		}

//		if(wordCounter % 10 == 0)
//		{
			//用于更新R矩阵
			for(j = V + hiddenSize; j < layer1Size; j ++)
			{
				hiddenInSyn[i*layer1Size + j].weight += bpttHiddenInSyn[i*layer1Size + j].weight - hiddenInSyn[i*layer1Size + j].weight * beta2;
				bpttHiddenInSyn[i*layer1Size + j].weight = 0;
			}
//		}
//		else
//		{
//			for(j = V + hiddenSize; j < layer1Size; j ++)
//			{
//				hiddenInSyn[i*layer1Size + j].weight += bpttHiddenInSyn[i*layer1Size + j].weight;
//				bpttHiddenInSyn[i*layer1Size + j].weight = 0;
//			}
//		}
	}

	if(!perSentUpdate)
	{
		// update parameters in sentences
		if(!conbptt)
			learnSent(senLen);
		else
			learnSentBPTT(senLen);
	}
}

void RNNPG::learnSentAdaGrad(int senLen)
{
	double beta2 = alpha * beta;
	int i, j, N = hiddenSize + hiddenSize;
	for(i = 0; i < hiddenSize; i ++)
		hisNeu[i].er *= hisNeu[i].ac * (1 - hisNeu[i].ac);
	clearNeurons(cmbNeu + hiddenSize, hiddenSize, 2);
	// back propagate error from the history representation to sentence top layer (the final representation of the sentence)
	matrixXvector(cmbNeu, hisNeu, compressSyn, hiddenSize + hiddenSize, 0, hiddenSize, hiddenSize, hiddenSize + hiddenSize, 1);
	// update compress matrix
//	if(wordCounter % 10 == 0)
//	{
		for(i = 0; i < hiddenSize; i ++)
			for(j = 0; j < N; j ++)
			{
				double grad = hisNeu[i].er * cmbNeu[j].ac;
				sumGradSquare.compressSyn_[i * N + j] += grad * grad;
				double move = alpha * grad * (sqrt(sumGradSquare.compressSyn_[i * N + j]) + adaGradEps);
				compressSyn[i * N + j].weight += move - compressSyn[i * N + j].weight * beta2;
			}
//	}
//	else
//	{
//		for(i = 0; i < hiddenSize; i ++)
//			for(j = 0; j < N; j ++)
//				compressSyn[i * N + j].weight += alpha * hisNeu[i].er * cmbNeu[j].ac;
//	}

	// error propagate in sentence model
	neuron **senNeu = senLen == 5 ? sen5Neu : sen7Neu;
	int SEN_HIGHT = senLen == 5 ? SEN5_HIGHT : SEN7_HIGHT;
	int unitNum = 1, unitNumNx = 1, a = -1, b = -1;
	for(i = 0; i < hiddenSize; i ++)
		senNeu[SEN_HIGHT - 1][i].er = cmbNeu[hiddenSize + i].er;
	for(i = SEN_HIGHT - 2; i >= 0; i --)
	{
		unitNumNx = unitNum + (conSeq[i] - 1);
		int offset = 0, offsetNx = 0, offsetCon = 0;
		// for readability, I just compute the deviation seperately
		for(a = 0; a < hiddenSize; a ++) for(b = 0; b < unitNum; b ++)
		{
			offset = a * unitNum;
			senNeu[i + 1][offset + b].er *= senNeu[i + 1][offset + b].ac * (1 - senNeu[i + 1][offset + b].ac);
		}

		// compute the back propagate error
		if(i != 0 || !fixSentenceModelFirstLayer)
		{
			for(a = 0; a < hiddenSize; a ++)
			{
				offset = a * unitNum;
				offsetNx = a * unitNumNx;
				offsetCon = a * conSeq[i];
				for(b = 0; b < unitNum; b ++)
				{
					for(j = 0; j < conSeq[i]; j ++)
						senNeu[i][offsetNx + b + j].er += senNeu[i + 1][offset + b].er * conSyn[i][offsetCon + j].weight;
				}
			}
			for(a = 0; a < hiddenSize; a ++)
			{
				offsetNx = a * unitNumNx;
				for(b = 0; b < unitNumNx; b ++)
				{
					//assert(senNeu[i][offsetNx + b].er >= -15 && senNeu[i][offsetNx + b].er <= 15);
					if(senNeu[i][offsetNx + b].er < -15) senNeu[i][offsetNx + b].er = -15;
					if(senNeu[i][offsetNx + b].er > 15) senNeu[i][offsetNx + b].er = 15;
				}
			}
		}

		// update the matrix, at this point we do NOT consider the L2 normalization term
		for(a = 0; a < hiddenSize; a ++)
		{
			offset = a * unitNum;
			offsetNx = a * unitNumNx;
			offsetCon = a * conSeq[i];
			for(j = 0; j < conSeq[i]; j ++)
				conSynOffset[i][offsetCon + j].weight = 0;
			for(b = 0; b < unitNum; b ++)
			{
				for(j = 0; j < conSeq[i]; j ++)
					conSynOffset[i][offsetCon + j].weight += alpha * senNeu[i + 1][offset + b].er * senNeu[i][offsetNx + b + j].ac;
			}
		}
		// update the matrix with L2 normalization term
		for(a = 0; a < hiddenSize; a ++)
		{
			offsetCon = a * conSeq[i];
			for(j = 0; j < conSeq[i]; j ++)
			{
				double grad = conSynOffset[i][offsetCon + j].weight / alpha;
				sumGradSquare.conSyn_[i][offsetCon + j] += grad * grad;
				double move = alpha * grad / (sqrt(sumGradSquare.conSyn_[i][offsetCon + j]) + adaGradEps);
				conSyn[i][offsetCon + j].weight += move - beta2 * conSyn[i][offsetCon + j].weight;
			}
		}

		unitNum = unitNumNx;
	}
	// cout << unitNumNx << endl;
	assert(unitNumNx == senLen);

	if(fixSentenceModelFirstLayer)
		return;
	int V = vocab.getVocabSize();
	int newWordCounter = wordCounter - senLen - 1;
	for(i = 0; i < unitNumNx; i ++)
	{
		int word = bpttHistory[i + 1];	// because bpttHistory recorded lastWord, not curWord
		newWordCounter ++;
		for(j = 0; j < hiddenSize; j ++)
		{
			senNeu[0][j*unitNumNx + i].er *= senNeu[0][j*unitNumNx + i].ac * (1 - senNeu[0][j*unitNumNx + i].ac);
			double grad = senNeu[0][j*unitNumNx + i].er;
			sumGradSquare.senweSyn_[V*j + word] += grad * grad;
			double move = alpha * grad / (sqrt(sumGradSquare.senweSyn_[V*j + word]) + adaGradEps);
			if(newWordCounter % 10 == 0)
				senweSyn[V*j + word].weight += move - beta2 * senweSyn[V*j + word].weight;
			else
				senweSyn[V*j + word].weight += move;
		}

//		if(newWordCounter % 10 == 0)
//		{
//			for(j = 0; j < hiddenSize; j ++)
//			{
//				senNeu[0][j*unitNumNx + i].er *= senNeu[0][j*unitNumNx + i].ac * (1 - senNeu[0][j*unitNumNx + i].ac);
//				senweSyn[V*j + word].weight += alpha * senNeu[0][j*unitNumNx + i].er - beta2 * senweSyn[V*j + word].weight;
//			}
//		}
//		else
//		{
//			for(j = 0; j < hiddenSize; j ++)
//			{
//				senNeu[0][j*unitNumNx + i].er *= senNeu[0][j*unitNumNx + i].ac * (1 - senNeu[0][j*unitNumNx + i].ac);
//				senweSyn[V*j + word].weight += alpha * senNeu[0][j*unitNumNx + i].er;
//			}
//		}
	}
}

void RNNPG::learnNetAdaGrad(int lastWord, int curWord, int wordPos, int senLen)
{
	double beta2 = alpha * beta;
	int curClassIndex = voc_arr[curWord].classIndex, i = 0, j = 0, V = vocab.getVocabSize(), N = 0, offset = 0;
	// error at output layer. 1. error on words
	for(i = classStart[curClassIndex]; i < classEnd[curClassIndex]; i ++)
		outNeu[i].er = 0 - outNeu[i].ac;
	outNeu[curWord].er = 1 - outNeu[curWord].ac;
	// error at output layer. 2. error on classes
	N = V + classSize;
	for(i = V; i < N; i ++)
		outNeu[i].er = 0 - outNeu[i].ac;
	outNeu[V + curClassIndex].er = 1 - outNeu[V + curClassIndex].ac;

	clearNeurons(hiddenNeu, hiddenSize, 2);

	// error backpropagation to hidden layer
	// 1. error from words to hidden
	matrixXvector(hiddenNeu, outNeu, outHiddenSyn, hiddenSize, classStart[curClassIndex], classEnd[curClassIndex], 0, hiddenSize, 1);
	// 2. error from classes to hidden
	matrixXvector(hiddenNeu, outNeu, outHiddenSyn, hiddenSize, V, V + classSize, 0, hiddenSize, 1);

	if(directError)
	{
		// bufOutConditionNeu
		clearNeurons(bufOutConditionNeu + (wordPos * hiddenSize), hiddenSize, 2);
		// error back propagate to conditionNeu
		// 1. error from words to conditionNeu
		matrixXvector(bufOutConditionNeu + (wordPos * hiddenSize), outNeu, outConditionDSyn, hiddenSize, classStart[curClassIndex], classEnd[curClassIndex], 0, hiddenSize, 1);
		// 2. error from classes to conditionNeu
		matrixXvector(bufOutConditionNeu + (wordPos * hiddenSize), outNeu, outConditionDSyn, hiddenSize, V, V + classSize, 0, hiddenSize, 1);
	}

	// updating the matrix outHiddenSyn, since we already have the error at output layer and the activation in the hidden layer
	// we update the weight per word rather than per sentence for faster increase in likelihood. Perhaps it will be modified to per sentence update later
	// update submatrix of words to hidden layer
	offset = classStart[curClassIndex] * hiddenSize;
	for(i = classStart[curClassIndex]; i < classEnd[curClassIndex]; i ++)
	{
		for(j = 0; j < hiddenSize; j ++)
		{
			double grad = outNeu[i].er * hiddenNeu[j].ac;
			sumGradSquare.outHiddenSyn_[offset + j] += grad * grad;
			double move = alpha * grad / (sqrt(sumGradSquare.outHiddenSyn_[offset + j]) + adaGradEps);

			if(wordCounter % 10 == 0)
				outHiddenSyn[offset + j].weight += move - beta2*outHiddenSyn[offset + j].weight;
			else
				outHiddenSyn[offset + j].weight += move;
		}
//		if(wordCounter % 10 == 0)
//			for(j = 0; j < hiddenSize; j ++) outHiddenSyn[offset + j].weight += alpha * outNeu[i].er * hiddenNeu[j].ac - beta2*outHiddenSyn[offset + j].weight;
//		else
//			for(j = 0; j < hiddenSize; j ++) outHiddenSyn[offset + j].weight += alpha * outNeu[i].er * hiddenNeu[j].ac;
		offset += hiddenSize;
	}
	// update submatrix of classes to hidden layer
	N = V + classSize;
	offset = V * hiddenSize;
	for(i = V; i < N; i ++)
	{
		for(j = 0; j < hiddenSize; j ++)
		{
			double grad = outNeu[i].er * hiddenNeu[j].ac;
			sumGradSquare.outHiddenSyn_[offset + j] += grad * grad;
			double move = alpha * grad / (sqrt(sumGradSquare.outHiddenSyn_[offset + j]) + adaGradEps);

			if(wordCounter % 10 == 0)
				outHiddenSyn[offset + j].weight += move - beta2*outHiddenSyn[offset + j].weight;
			else
				outHiddenSyn[offset + j].weight += move;
		}
//		if(wordCounter % 10 == 0)
//			for(j = 0; j < hiddenSize; j ++) outHiddenSyn[offset + j].weight += alpha * outNeu[i].er * hiddenNeu[j].ac - beta2*outHiddenSyn[offset + j].weight;
//		else
//			for(j = 0; j < hiddenSize; j ++) outHiddenSyn[offset + j].weight += alpha * outNeu[i].er * hiddenNeu[j].ac;
		offset += hiddenSize;
	}

	if(directError)
	{
		// update the matrix outConditionDSyn
		// update submatrix of words to condition layer
		offset = classStart[curClassIndex] * hiddenSize;
		for(i = classStart[curClassIndex]; i < classEnd[curClassIndex]; i ++)
		{
			// how can I make such stupid bug !!! hiddenNeu[V + j].ac is obviously incorrect !!! inNeu[V + j].ac
			if(wordCounter % 10 == 0)
				for(j = 0; j < hiddenSize; j ++) outConditionDSyn[offset + j].weight += alpha * outNeu[i].er * inNeu[V + j].ac - beta2*outConditionDSyn[offset + j].weight;
			else
				for(j = 0; j < hiddenSize; j ++) outConditionDSyn[offset + j].weight += alpha * outNeu[i].er * inNeu[V + j].ac;
			offset += hiddenSize;
		}

		// update submatrix of classes to condition layer
		N = V + classSize;
		offset = V * hiddenSize;
		for(i = V; i < N; i ++)
		{
			// how can I make such stupid bug !!! hiddenNeu[V + j].ac is obviously incorrect !!! inNeu[V + j].ac
			if(wordCounter % 10 == 0)
				for(j = 0; j < hiddenSize; j ++) outConditionDSyn[offset + j].weight += alpha * outNeu[i].er * inNeu[V + j].ac - beta2*outConditionDSyn[offset + j].weight;
			else
				for(j = 0; j < hiddenSize; j ++) outConditionDSyn[offset + j].weight += alpha * outNeu[i].er * inNeu[V + j].ac;
			offset += hiddenSize;
		}
	}

	// this is for back propagation through time
	bpttHistory[wordPos] = lastWord;	// store the last word
	memcpy(bpttHiddenNeu + (wordPos * hiddenSize), hiddenNeu, sizeof(neuron)*hiddenSize);	// store the hidden layer
	memcpy(bpttInHiddenNeu + (wordPos * hiddenSize), inNeu + (V + hiddenSize), sizeof(neuron)*hiddenSize);	// store the hidden units in input layer (previous hidden layer)
	memcpy(bpttConditionNeu + (wordPos * hiddenSize), inNeu + V, sizeof(neuron)*hiddenSize);	// store the condition units in input layer
	if(curWord != 0)
		return;
	// if this is the end of sentence, then let's do it
	int lword = -1, layer1Size = V + hiddenSize + hiddenSize;
	synapse **mapSyn = NULL;
	mapSyn = senLen == 5 ? map5Syn : map7Syn;
	double **mapSyn_ = senLen == 5 ? sumGradSquare.map5Syn_ : sumGradSquare.map7Syn_;
	for(int step = wordPos; step >= 0; step --)
	{
		// take care of vocabulary and recurrent part in input layer first
		// bpttHiddenInSyn[]
		for(i = 0; i < hiddenSize; i ++)
			hiddenNeu[i].er *= hiddenNeu[i].ac * (1 - hiddenNeu[i].ac);
		lword = bpttHistory[step];
		// accumulate deviations for input matrix, X
		for(i = 0; i < hiddenSize; i ++)
			bpttHiddenInSyn[i * layer1Size + lword].weight += alpha * hiddenNeu[i].er;

		clearNeurons(inNeu + (V+hiddenSize), hiddenSize, 2);
		matrixXvector(inNeu, hiddenNeu, hiddenInSyn, layer1Size, 0, hiddenSize, V + hiddenSize, layer1Size, 1);
		// accumulate deviations for hidden matrix, R
		for(i = 0; i < hiddenSize; i ++)
			for(j = V + hiddenSize; j < layer1Size; j ++)
				bpttHiddenInSyn[i*layer1Size + j].weight += alpha * hiddenNeu[i].er * inNeu[j].ac;

		// now we take care the condition part in the input layer
		// back propagate the error to condition part
		clearNeurons(inNeu + V, hiddenSize, 2);
		matrixXvector(inNeu, hiddenNeu, hiddenInSyn, layer1Size, 0, hiddenSize, V, V + hiddenSize, 1);
		// accumulate deviations for condition matrix, H
		N = V + hiddenSize;
		for(i = 0; i < hiddenSize; i ++)
			for(j = V; j < N; j ++)
				bpttHiddenInSyn[i*layer1Size + j].weight += alpha * hiddenNeu[i].er * inNeu[j].ac;

		if(directError)
		{
			for(i = 0; i < hiddenSize; i ++)
				inNeu[V + i].er += bufOutConditionNeu[step * hiddenSize + i].er;
		}

		for(i = V; i < N; i ++)
			inNeu[i].er *= inNeu[i].ac * (1 - inNeu[i].ac);

		if(perSentUpdate)
			clearNeurons(hisNeu, hiddenSize, 2);

		// watch that the error in hisNeu must be inilizated to zero at the beginning of dealing with each sentence
		matrixXvector(hisNeu, inNeu + V, mapSyn[step], hiddenSize, 0, hiddenSize, 0, hiddenSize, 1);

		// acumulate deviations for map matrix
		for(i = 0; i < hiddenSize; i ++)
			for(j = 0; j < hiddenSize; j ++)
			{
				double grad = inNeu[V+i].er * hisNeu[j].ac;
				mapSyn_[step][i*hiddenSize + j] += grad * grad;
				double move = alpha * grad / (sqrt(mapSyn_[step][i*hiddenSize + j]) + adaGradEps);
				if(wordCounter % 10 == 0)
					mapSyn[step][i * hiddenSize + j].weight += move - mapSyn[step][i * hiddenSize + j].weight * beta2;
				else
					mapSyn[step][i * hiddenSize + j].weight += move;
			}
//		if(wordCounter % 10 == 0)
//		{
//			for(i = 0; i < hiddenSize; i ++)
//				for(j = 0; j < hiddenSize; j ++)
//					mapSyn[step][i * hiddenSize + j].weight += alpha * inNeu[V+i].er * hisNeu[j].ac - mapSyn[step][i * hiddenSize + j].weight * beta2;
//		}
//		else
//		{
//			for(i = 0; i < hiddenSize; i ++)
//				for(j = 0; j < hiddenSize; j ++)
//					mapSyn[step][i * hiddenSize + j].weight += alpha * inNeu[V+i].er * hisNeu[j].ac;
//		}

		if(perSentUpdate)
			learnSent(senLen);

		if(step == 0) continue;
		// propagate error to previous layer
		for(i = 0; i < hiddenSize; i ++)
		{
			hiddenNeu[i].er = inNeu[V + hiddenSize + i].er + bpttHiddenNeu[(step - 1)*hiddenSize + i].er;
			hiddenNeu[i].ac = bpttHiddenNeu[(step - 1)*hiddenSize + i].ac;
		// restore the recurrent part in input layer
			inNeu[V + hiddenSize + i].ac = bpttInHiddenNeu[(step - 1)*hiddenSize + i].ac;
		// restore the condition part in input layer
			inNeu[V + i].ac = bpttConditionNeu[(step - 1)*hiddenSize + i].ac;
		}
	}

	// restore hidden layer
	memcpy(hiddenNeu, bpttHiddenNeu + (wordPos * hiddenSize), sizeof(neuron)*hiddenSize);


	// update input matrix, X, condition Matrix, H and recurrent matrix, R
	for(i = 0; i < hiddenSize; i ++)
	{
//		if(wordCounter % 10 == 0)
//		{
			for(j = 0; j <= wordPos; j ++)
			{
				lword = bpttHistory[j];
				double grad = bpttHiddenInSyn[i*layer1Size + lword].weight / alpha;
				sumGradSquare.hiddenInSyn_[i*layer1Size + lword] += grad * grad;
				double move = alpha * grad / (sqrt(sumGradSquare.hiddenInSyn_[i*layer1Size + lword]) + adaGradEps);
				hiddenInSyn[i*layer1Size + lword].weight += move - hiddenInSyn[i*layer1Size + lword].weight * beta2;
				bpttHiddenInSyn[i*layer1Size + lword].weight = 0;
			}
//		}
//		else
//		{
//			for(j = 0; j <= wordPos; j ++)
//			{
//				lword = bpttHistory[j];
//				hiddenInSyn[i*layer1Size + lword].weight += bpttHiddenInSyn[i * layer1Size + lword].weight;
//				bpttHiddenInSyn[i * layer1Size + lword].weight = 0;
//			}
//		}

//		if(wordCounter % 10 == 0)
//		{
			N = V + hiddenSize;
			for(j = V; j < N; j ++)
			{
				double grad = bpttHiddenInSyn[i*layer1Size + j].weight / alpha;
				sumGradSquare.hiddenInSyn_[i*layer1Size + j] += grad * grad;
				double move = alpha * grad / (sqrt(sumGradSquare.hiddenInSyn_[i*layer1Size + j]) + adaGradEps);
				hiddenInSyn[i*layer1Size + j].weight += move - hiddenInSyn[i*layer1Size + j].weight * beta2;
				bpttHiddenInSyn[i*layer1Size + j].weight = 0;
			}
//		}
//		else
//		{
//			N = V + hiddenSize;
//			for(j = V; j < N; j ++)
//			{
//				hiddenInSyn[i*layer1Size + j].weight += bpttHiddenInSyn[i*layer1Size + j].weight;
//				bpttHiddenInSyn[i*layer1Size + j].weight = 0;
//			}
//		}

//		if(wordCounter % 10 == 0)
//		{
			for(j = V + hiddenSize; j < layer1Size; j ++)
			{
				double grad = bpttHiddenInSyn[i*layer1Size + j].weight / alpha;
				sumGradSquare.hiddenInSyn_[i*layer1Size + j] += grad * grad;
				double move = alpha * grad / (sqrt(sumGradSquare.hiddenInSyn_[i*layer1Size + j]) + adaGradEps);
				hiddenInSyn[i*layer1Size + j].weight += move - hiddenInSyn[i*layer1Size + j].weight * beta2;
				bpttHiddenInSyn[i*layer1Size + j].weight = 0;
			}
//		}
//		else
//		{
//			for(j = V + hiddenSize; j < layer1Size; j ++)
//			{
//				hiddenInSyn[i*layer1Size + j].weight += bpttHiddenInSyn[i*layer1Size + j].weight;
//				bpttHiddenInSyn[i*layer1Size + j].weight = 0;
//			}
//		}
	}

//	if(!perSentUpdate)
//	{
//		// update parameters in sentences
//		if(!conbptt)
//			learnSent(senLen);
//		else
//			learnSentBPTT(senLen);
//	}
	learnSentAdaGrad(senLen);
}

/**
 * @brief
 * 将r_{j-1}的ac拷贝到r_j中
 */
void RNNPG::copyHiddenLayerToInput()
{
	int offset = vocab.getVocabSize() + hiddenSize;
	for(int i = 0; i < hiddenSize; i ++)
		inNeu[offset + i].ac = hiddenNeu[i].ac;
}

/**
 * @brief
 * 使用一首诗进行训练
 * @param sentences 存储一首诗里的每句话的向量
 */
void RNNPG::trainPoem(const vector<string> &sentences)
{
	//判断5言诗还是7言诗
	const int SEN_NUM = 4;
	vector<string> words;//每个字占用vector里的一个空间
	int i, SEN_HIGHT = -1;
	neuron **senNeu = NULL;
	words.clear();
	split(sentences[0].c_str(), " ", words);
	SEN_HIGHT = words.size() == 5 ? SEN5_HIGHT : SEN7_HIGHT;
	senNeu = words.size() == 5 ? sen5Neu : sen7Neu;

	// this is for the first sentence
	//初始化CSM网络
	initSent(words.size());
	//计算一个句子的表达
	neuron *sen_repr = sen2vec(words, senNeu, SEN_HIGHT);		// this is the pointer for the top layer sentence model, DO NOT modify it

	// for first sentence, we can just give the representation to the generation model, or
	clearNeurons(cmbNeu, hiddenSize * 2, 3);		// 这个好像是早期的注释，这段代码看上去没有问题，this is probably a bug!!! change 1 to 3, also flush the error

	// init activation in recurrent context model，对RCM进行初始计算
	// h_i=\sigma (M\cdot \begin{bmatrix}h_{i-1}\\v_i\end{bmatrix})
	// 使用historyStableAC对cmbNeu中的h_{i-1}进行初始化
	for(i = 0; i < hiddenSize; i ++)
		cmbNeu[i].ac = historyStableAC;
	memcpy(cmbNeu + hiddenSize, sen_repr, sizeof(neuron)*hiddenSize);
	clearNeurons(hisNeu, hiddenSize, 3);
	matrixXvector(hisNeu, cmbNeu, compressSyn, hiddenSize * 2, 0, hiddenSize, 0, hiddenSize * 2, 0);
	//激活
	funACNeurons(hisNeu, hiddenSize);

	// alternivate
	// memcpy(hisNeu, sen_repr, sizeof(neuron)*hiddenSize);

	synapse **mapSyn = words.size() == 5 ? map5Syn : map7Syn;
	// this is for the subsequence sentences (generation and compress the representation)
	//现在从第1句开始计算，之前是计算的第0句
	for(i = 1; i < SEN_NUM; i ++)
	{
		isLastSentOfPoem = i == SEN_NUM - 1;
		contextBPTTSentNum = i;
		words.clear();
		//取出一首诗的第i句放到words这个容器里
		split(sentences[i].c_str(), " ", words);
		// just for test...
		// printNeurons(hisNeu, hiddenSize);

		// generation ...
		// initSent(words.size());
		if(flushOption == EVERY_SENTENCE)
			flushNet();		// clear input hidden and output layer
		else if(flushOption == EVERY_POEM)
		{
			if(i == 1)
				flushNet();
		}

		words.push_back("</s>");	// during generation, we DO care about the End-of-Sentence，注意在这里添加了一个结尾的符号，因此实际上一个句子的长度是words.size() - 1
		int lastWord = 0, curWord = -1, wdPos;
		//wdPos是当前正在处理的词的下标，lastWord指的诗上一个处理的词对应的词的编号，curWord指的是当前正在处理的词对应的词的编号
		for(wdPos = 0; wdPos < (int)words.size(); wdPos ++)
		{
			wordCounter ++;
			curWord = vocab.getVocabID(words[wdPos].c_str());
			if(curWord == -1)
				cout << "unseen word " << "'" << words[wdPos] << "'" << endl;
			//在训练过程中是不可能遇到没有见到过的新词的
			assert(curWord != -1);		// this is impossible, or there is a bug!
			inNeu[lastWord].ac = 1;
			computeNet(lastWord, curWord, wdPos, mapSyn);
			// perhaps I also need to caculate the log-likelihood
			// 服从以下假设P(word,word_context|context)=P(word|word_class,context) \cdot P(word_class|context)
			logp+=log10(outNeu[voc_arr[curWord].classIndex+vocab.getVocabSize()].ac * outNeu[curWord].ac);
			// learnNet, tomorrow come back to the sentence model
			if(!adaGrad)
				learnNet(lastWord, curWord, wdPos, words.size() - 1);
			else
				learnNetAdaGrad(lastWord, curWord, wdPos, words.size() - 1);
			inNeu[lastWord].ac = 0;
			copyHiddenLayerToInput();
			lastWord = curWord;
		}
		words.pop_back();	// generation done and delete </s>

		// compress representation
		if(i == SEN_NUM - 1)
			// 如果已经训练到了最后一句，就停止循环
			break;
		//如果没有训练到最后一句，进行下一句诗的句子的表达的计算
		initSent(words.size());
		sen_repr = sen2vec(words, senNeu, SEN_HIGHT);
		//更新$\begin{bmatrix}v_i\\h_{i-1}\end{bmatrix}$
		memcpy(cmbNeu, hisNeu, sizeof(neuron)*hiddenSize);
		memcpy(cmbNeu + hiddenSize, sen_repr, sizeof(neuron)*hiddenSize);
		clearNeurons(hisNeu, hiddenSize, 3);
		//计算h_i
		matrixXvector(hisNeu, cmbNeu, compressSyn, hiddenSize * 2, 0, hiddenSize, 0, hiddenSize * 2, 0);
		funACNeurons(hisNeu, hiddenSize);
	}
}

/**
 * @brief
 * 使用一首诗进行测试
 * @param sentences 存储一首诗里的每句话的向量
 */
void RNNPG::testPoem(const vector<string> &sentences)
{
	const int SEN_NUM = 4;
	vector<string> words;
	int i, SEN_HIGHT = -1;
	//CSM中对应每句诗的神经元
	neuron **senNeu = NULL;
	words.clear();
	split(sentences[0].c_str(), " ", words);
	SEN_HIGHT = words.size() == 5 ? SEN5_HIGHT : SEN7_HIGHT;
	senNeu = words.size() == 5 ? sen5Neu : sen7Neu;

	// this is for the first sentence
	initSent(words.size());
	neuron *sen_repr = sen2vec(words, senNeu, SEN_HIGHT);		// this is the pointer for the top layer sentence model, DO NOT modify it

	// for first sentence, we can just give the representation to the generation model, or
	clearNeurons(cmbNeu, hiddenSize * 2, 3);		// 这个好像是早期的bug，现在看上去好像没有什么问题，this is probably a bug!!! change 1 to 3, also flush the error
	//和训练的时候不一样，这里没有设置h_0为historyStableAC
	memcpy(cmbNeu + hiddenSize, sen_repr, sizeof(neuron)*hiddenSize);
	clearNeurons(hisNeu, hiddenSize, 3);
	matrixXvector(hisNeu, cmbNeu, compressSyn, hiddenSize * 2, 0, hiddenSize, 0, hiddenSize * 2, 0);
	funACNeurons(hisNeu, hiddenSize);

	// alternivate
	// memcpy(hisNeu, sen_repr, sizeof(neuron)*hiddenSize);

	synapse **mapSyn = words.size() == 5 ? map5Syn : map7Syn;
	// this is for the subsequence sentences (generation and compress the representation)
	//现在从第1句开始计算，之前是计算的第0句
	for(i = 1; i < SEN_NUM; i ++)
	{
		words.clear();
		split(sentences[i].c_str(), " ", words);
		// just for test...
		// printNeurons(hisNeu, hiddenSize);

		// generation ...
		// initSent(words.size());
		// flushNet();		// clear input hidden and output layer
		if(flushOption == EVERY_SENTENCE)
			flushNet();		// clear input hidden and output layer
		else if(flushOption == EVERY_POEM)
		{
			if(i == 1)
				flushNet();
		}
		words.push_back("</s>");	// during generation, we DO care about the End-of-Sentence
		int lastWord = 0, curWord = -1, wdPos;
		for(wdPos = 0; wdPos < (int)words.size(); wdPos ++)
		{
			wordCounter ++;
			curWord = vocab.getVocabID(words[wdPos].c_str());
			bool isRare = false;
			if(curWord == -1)
			{
				//在测试过程中可能遇到新词,对于出现的新词，word embedding矩阵L和X的相应位置应该都是初始化的默认值
				// when the word cannot be found, we use <R> instead
				curWord = vocab.getVocabID("<R>");
				isRare = true;
			}
			//表明绝对不能出现curWord是-1，它要被替换成<R>对应的id
			assert(curWord != -1);		// this is impossible, or there is a bug!
			inNeu[lastWord].ac = 1;
			computeNet(lastWord, curWord, wdPos, mapSyn);
			// perhaps I also need to caculate the log-likelihood
			if(!isRare)
				// 服从以下假设P(word,word_context|context)=P(word|word_class,context) \cdot P(word_class|context)
				logp += log10(outNeu[voc_arr[curWord].classIndex+vocab.getVocabSize()].ac * outNeu[curWord].ac);
			else
				// voc_arr[curWord].freq指的是<R>这个未见词出现的次数
				logp += log10(outNeu[voc_arr[curWord].classIndex+vocab.getVocabSize()].ac * outNeu[curWord].ac / voc_arr[curWord].freq);
			// learnNet, tomorrow come back to the sentence model
			// learnNet(lastWord, curWord, wdPos, words.size() - 1);
			inNeu[lastWord].ac = 0;
			copyHiddenLayerToInput();
			lastWord = curWord;
		}
		words.pop_back();	// generation done and delete </s>

		// compress representation
		if(i == SEN_NUM - 1)
			// 如果已经训练到了最后一句，就停止循环
			break;
		//如果没有训练到最后一句，进行下一句诗的句子的表达的计算
		initSent(words.size());
		sen_repr = sen2vec(words, senNeu, SEN_HIGHT);
		memcpy(cmbNeu, hisNeu, sizeof(neuron)*hiddenSize);
		memcpy(cmbNeu + hiddenSize, sen_repr, sizeof(neuron)*hiddenSize);
		clearNeurons(hisNeu, hiddenSize, 3);
		matrixXvector(hisNeu, cmbNeu, compressSyn, hiddenSize * 2, 0, hiddenSize, 0, hiddenSize * 2, 0);
		funACNeurons(hisNeu, hiddenSize);
	}
}

void RNNPG::initBackup()
{
	int i = -1, M = -1, N = -1, unitNum;
	M = MAX_CON_N;
	for(i = 0; i < M; i ++)
	{
		N = hiddenSize * conSeq[i];
		conSyn_backup[i] = (synapse*)xmalloc(N*sizeof(synapse));
	}
	for(i = 0; i < M; i ++)
	{
		N = hiddenSize * conSeq[i];
		conSynOffset_backup[i] = (synapse*)xmalloc(N*sizeof(synapse));
	}
	M = SEN7_HIGHT;
	unitNum = 7;
	for(i = 0; i < M; i ++)
	{
		N = hiddenSize * unitNum;
		sen7Neu_backup[i] = (neuron*)xmalloc(N * sizeof(neuron));
		if(unitNum > 1)	unitNum -= conSeq[i] - 1;
	}
	M = SEN5_HIGHT;
	unitNum = 5;
	for(i = 0; i < M; i ++)
	{
		N = hiddenSize * unitNum;
		sen5Neu_backup[i] = (neuron*)xmalloc(N * sizeof(neuron));
		if(unitNum > 1)	unitNum -= conSeq[i] - 1;
	}
	M = hiddenSize * hiddenSize * 2;
	compressSyn_backup = (synapse*)xmalloc(M * sizeof(synapse));
	M = hiddenSize;
	hisNeu_backup = (neuron*)xmalloc(M * sizeof(neuron));
	M = hiddenSize * 2;
	cmbNeu_backup = (neuron*)xmalloc(M * sizeof(neuron));
	M = 8; N = hiddenSize * hiddenSize;
	for(i = 0; i < M; i ++)
	{
		map7Syn_backup[i] = (synapse*)xmalloc(N * sizeof(synapse));
		// map7Syn_backup[i][j].weight = map7Syn[i][j].weight;
	}
	M = 6; N = hiddenSize * hiddenSize;
	for(i = 0; i < M; i ++)
	{
		map5Syn_backup[i] = (synapse*)xmalloc(N * sizeof(synapse));
		// map5Syn_backup[i][j].weight = map5Syn[i][j].weight;
	}
	M = hiddenSize;
	conditionNeu_backup = (neuron*)xmalloc(M * sizeof(neuron));
	M = vocab.getVocabSize() * hiddenSize;
	senweSyn_backup = (synapse*)xmalloc(M * sizeof(synapse));
	M = vocab.getVocabSize() + hiddenSize + hiddenSize;
	inNeu_backup = (neuron*)xmalloc(M * sizeof(neuron));
	M = hiddenSize;
	hiddenNeu_backup = (neuron*)xmalloc(M * sizeof(neuron));
	M = vocab.getVocabSize() + classSize;
	outNeu_backup = (neuron*)xmalloc(M * sizeof(neuron));
	M = hiddenSize * (vocab.getVocabSize() + hiddenSize + hiddenSize);
	hiddenInSyn_backup = (synapse*)xmalloc(M * sizeof(synapse));
	M = (vocab.getVocabSize() + classSize) * hiddenSize;
	outHiddenSyn_backup = (synapse*)xmalloc(M * sizeof(synapse));

	M = ((vocab.getVocabSize() + classSize) * hiddenSize);
	outConditionDSyn_backup = (synapse*)xmalloc(M * sizeof(synapse));
}

void RNNPG::saveWeights()
{
	int i = -1, j = -1, M = -1, N = -1, unitNum;
	M = MAX_CON_N;
	for(i = 0; i < M; i ++)
	{
		N = hiddenSize * conSeq[i];
		for(j = 0; j < N; j ++)
			conSyn_backup[i][j].weight = conSyn[i][j].weight;
	}
	for(i = 0; i < M; i ++)
	{
		N = hiddenSize * conSeq[i];
		for(j = 0; j < N; j ++)
		{
			conSynOffset_backup[i][j].weight = conSynOffset[i][j].weight;
		}
	}
	M = SEN7_HIGHT;
	unitNum = 7;
	for(i = 0; i < M; i ++)
	{
		N = hiddenSize * unitNum;
		for(j = 0; j < N; j ++)
		{
			sen7Neu_backup[i][j].ac = sen7Neu[i][j].ac;
			sen7Neu_backup[i][j].er = sen7Neu[i][j].er;
		}
		if(unitNum > 1)	unitNum -= conSeq[i] - 1;
	}
	M = SEN5_HIGHT;
	unitNum = 5;
	for(i = 0; i < M; i ++)
	{
		N = hiddenSize * unitNum;
		for(j = 0; j < N; j ++)
		{
			sen5Neu_backup[i][j].ac = sen5Neu[i][j].ac;
			sen5Neu_backup[i][j].er = sen5Neu[i][j].er;
		}
		if(unitNum > 1)	unitNum -= conSeq[i] - 1;
	}
	M = hiddenSize * hiddenSize * 2;
	for(i = 0; i < M; i ++)
	{
		compressSyn_backup[i].weight = compressSyn[i].weight;
	}
	M = hiddenSize;
	for(i = 0; i < M; i ++)
	{
		hisNeu_backup[i].ac = hisNeu[i].ac;
		hisNeu_backup[i].er = hisNeu[i].er;
	}
	M = hiddenSize * 2;
	for(i = 0; i < M; i ++)
	{
		cmbNeu_backup[i].ac = cmbNeu[i].ac;
		cmbNeu_backup[i].er = cmbNeu[i].er;
	}
	M = 8; N = hiddenSize * hiddenSize;
	for(i = 0; i < M; i ++) for(j = 0; j < N; j ++)
	{
		map7Syn_backup[i][j].weight = map7Syn[i][j].weight;
	}
	M = 6; N = hiddenSize * hiddenSize;
	for(i = 0; i < M; i ++) for(j = 0; j < N; j ++)
	{
		map5Syn_backup[i][j].weight = map5Syn[i][j].weight;
	}
	M = hiddenSize;
	for(i = 0; i < M; i ++)
	{
		conditionNeu_backup[i].ac = conditionNeu[i].ac;
		conditionNeu_backup[i].er = conditionNeu[i].er;
	}
	M = vocab.getVocabSize() * hiddenSize;
	for(i = 0; i < M; i ++)
	{
		senweSyn_backup[i].weight = senweSyn[i].weight;
	}
	M = vocab.getVocabSize() + hiddenSize + hiddenSize;
	for(i = 0; i < M; i ++)
	{
		inNeu_backup[i].ac = inNeu[i].ac;
		inNeu_backup[i].er = inNeu[i].er;
	}
	M = hiddenSize;
	for(i = 0; i < M; i ++)
	{
		hiddenNeu_backup[i].ac = hiddenNeu[i].ac;
		hiddenNeu_backup[i].er = hiddenNeu[i].er;
	}
	M = vocab.getVocabSize() + classSize;
	for(i = 0; i < M; i ++)
	{
		outNeu_backup[i].ac = outNeu[i].ac;
		outNeu_backup[i].er = outNeu[i].er;
	}
	M = hiddenSize * (vocab.getVocabSize() + hiddenSize + hiddenSize);
	for(i = 0; i < M; i ++)
	{
		hiddenInSyn_backup[i].weight = hiddenInSyn[i].weight;
	}
	M = (vocab.getVocabSize() + classSize) * hiddenSize;
	for(i = 0; i < M; i ++)
	{
		outHiddenSyn_backup[i].weight = outHiddenSyn[i].weight;
	}
	M = (vocab.getVocabSize() + classSize) * hiddenSize;
	for(i = 0; i < M; i ++)
		outConditionDSyn_backup[i].weight = outConditionDSyn[i].weight;
}

void RNNPG::restoreWeights()
{
	int i = -1, j = -1, M = -1, N = -1, unitNum;
	M = MAX_CON_N;
	for(i = 0; i < M; i ++)
	{
		N = hiddenSize * conSeq[i];
		for(j = 0; j < N; j ++)
			conSyn[i][j].weight = conSyn_backup[i][j].weight;
	}
	for(i = 0; i < M; i ++)
	{
		N = hiddenSize * conSeq[i];
		for(j = 0; j < N; j ++)
		{
			conSynOffset[i][j].weight = conSynOffset_backup[i][j].weight;
		}
	}
	M = SEN7_HIGHT;
	unitNum = 7;
	for(i = 0; i < M; i ++)
	{
		N = hiddenSize * unitNum;
		for(j = 0; j < N; j ++)
		{
			sen7Neu[i][j].ac = sen7Neu_backup[i][j].ac;
			sen7Neu[i][j].er = sen7Neu_backup[i][j].er;
		}
		if(unitNum > 1)	unitNum -= conSeq[i] - 1;
	}
	M = SEN5_HIGHT;
	unitNum = 5;
	for(i = 0; i < M; i ++)
	{
		N = hiddenSize * unitNum;
		for(j = 0; j < N; j ++)
		{
			sen5Neu[i][j].ac = sen5Neu_backup[i][j].ac;
			sen5Neu[i][j].er = sen5Neu_backup[i][j].er;
		}
		if(unitNum > 1)	unitNum -= conSeq[i] - 1;
	}
	M = hiddenSize * hiddenSize * 2;
	for(i = 0; i < M; i ++)
	{
		compressSyn[i].weight = compressSyn_backup[i].weight;
	}
	M = hiddenSize;
	for(i = 0; i < M; i ++)
	{
		hisNeu[i].ac = hisNeu_backup[i].ac;
		hisNeu[i].er = hisNeu_backup[i].er;
	}
	M = hiddenSize * 2;
	for(i = 0; i < M; i ++)
	{
		cmbNeu[i].ac = cmbNeu_backup[i].ac;
		cmbNeu[i].er = cmbNeu_backup[i].er;
	}
	M = 8; N = hiddenSize * hiddenSize;
	for(i = 0; i < M; i ++) for(j = 0; j < N; j ++)
	{
		map7Syn[i][j].weight = map7Syn_backup[i][j].weight;
	}
	M = 6; N = hiddenSize * hiddenSize;
	for(i = 0; i < M; i ++) for(j = 0; j < N; j ++)
	{
		map5Syn[i][j].weight = map5Syn_backup[i][j].weight;
	}
	M = hiddenSize;
	for(i = 0; i < M; i ++)
	{
		conditionNeu[i].ac = conditionNeu_backup[i].ac;
		conditionNeu[i].er = conditionNeu_backup[i].er;
	}
	M = vocab.getVocabSize() * hiddenSize;
	for(i = 0; i < M; i ++)
	{
		senweSyn[i].weight = senweSyn_backup[i].weight;
	}
	M = vocab.getVocabSize() + hiddenSize + hiddenSize;
	for(i = 0; i < M; i ++)
	{
		inNeu[i].ac = inNeu_backup[i].ac;
		inNeu[i].er = inNeu_backup[i].er;
	}
	M = hiddenSize;
	for(i = 0; i < M; i ++)
	{
		hiddenNeu[i].ac = hiddenNeu_backup[i].ac;
		hiddenNeu[i].er = hiddenNeu_backup[i].er;
	}
	M = vocab.getVocabSize() + classSize;
	for(i = 0; i < M; i ++)
	{
		outNeu[i].ac = outNeu_backup[i].ac;
		outNeu[i].er = outNeu_backup[i].er;
	}
	M = hiddenSize * (vocab.getVocabSize() + hiddenSize + hiddenSize);
	for(i = 0; i < M; i ++)
	{
		hiddenInSyn[i].weight = hiddenInSyn_backup[i].weight;
	}
	M = (vocab.getVocabSize() + classSize) * hiddenSize;
	for(i = 0; i < M; i ++)
	{
		outHiddenSyn[i].weight = outHiddenSyn_backup[i].weight;
	}
	M = (vocab.getVocabSize() + classSize) * hiddenSize;
	for(i = 0; i < M; i ++)
		outConditionDSyn[i].weight = outConditionDSyn_backup[i].weight;
}

/**
 * @brief
 * 使用一个保存诗的文件进行测试
 * @param testF 文件路径
 */
void RNNPG::testNetFile(const char *testF)
{
	FILE *fin = xfopen(testF, "r", "computeNet -- open valid/testFile");
	char buf[1024];
	vector<string> sentences;
	int SEN_NUM = 4;
	flushNet();
	while(fgets(buf, sizeof(buf), fin))
	{
		sentences.clear();
		split(buf, "\t\r\n", sentences);
		if((int)sentences.size() != SEN_NUM) // here is just for quatrain
		{
			fprintf(stderr, "This is NOT a quatrain!!!\n");
			continue;
		}
		testPoem(sentences);
	}
	fclose(fin);
}

/**
 * @brief
 * 训练整个网络
 */
void RNNPG::trainNet()
{
	mode = TRAIN_MODE;

	loadVocab(trainFile);
	//初始化
	initNet();
	showParameters();

	double oriAlpha = alpha;

	char buf[1024];		// for poems this is enough
	vector<string> sentences;//存储一首诗里的每一行
	const int SEN_NUM = 4;
	double lastLogp = -1e18;
	int iter;
	//maxIter是最大的迭代次数
	for(iter = 0; iter < maxIter; iter ++)
	{
		if(adaGrad)
			sumGradSquare.reset(this);
		logp = 0;
		wordCounter = 0;
		//打开训练文件
		FILE *fin = xfopen(trainFile, "r", "computeNet -- open trainFile");
		int poem_cnt = 0;
		flushNet();		// for each interation, flush the net first
		while(fgets(buf,sizeof(buf),fin))	// one line is one poem, and one line will NOT exceed 1023 chars
		{
			sentences.clear();
			split(buf, "\t\r\n", sentences);
			//判断读取的是不是绝句，该模型只能对绝句进行处理
			if((int)sentences.size() != SEN_NUM) // here is just for quatrain
			{
				fprintf(stderr, "This is NOT a quatrain!!!\n");
				continue;
			}

			//这里是训练的部分
			trainPoem(sentences);

			poem_cnt ++;
			//每训练完100首诗的时候打印训练的结果
			if(poem_cnt % 100 == 0)
			{
				printf("%cIter: %3d\tAlpha: %f\t   TRAIN entropy: %.4f (%.4f)   Progress: %.2f%%", 13, iter, alpha, -logp/log10(2)/wordCounter,
						exp10(-logp/(double)wordCounter), poem_cnt/(double)totalPoemCount*100);
				fflush(stdout);
			}

			// just for observation
			if(saveModel == 1)
			{
				if(poem_cnt % 10000 == 1)
				{
					char modelName[1024];
					sprintf(modelName, "%s_iter%03d_%05d", modelFile, iter, poem_cnt);
					saveNet(modelName);
				}
			}
		}
		fclose(fin);
		//ascii中13对应回车
		printf("%cIter: %3d\tAlpha: %f\t   TRAIN entropy: %.4f (%.4f)    ", 13, iter, alpha, -logp/log10(2)/wordCounter, exp10(-logp/(double)wordCounter));
		fflush(stdout);

		// this is for the validation data
		logp = 0;
		wordCounter = 0;
		flushNet();
		testNetFile(validFile);
		printf("VALID entropy: %.4f (%.4f)    ", -logp/log10(2)/wordCounter, exp10(-logp/(double)wordCounter));

		double validationLogp = logp;

		// this is for the test data
		logp = 0;
		wordCounter = 0;
		flushNet();
		testNetFile(testFile);
		printf("TEST entropy: %.4f (%.4f)\n", -logp/log10(2)/wordCounter, exp10(-logp/(double)wordCounter));

		logp = validationLogp;

		if (logp*minImprovement < lastLogp)
			restoreWeights();
		else
			saveWeights();

		//对数似然*最小的进步小于lastLogp就停止训练，换句话说，就是训练过程中误差的改变已经比较小了，这个时候有两部，一是降低学习率，如果已经降低之后模型得到的改进还是很小，那么停止训练
		if (logp*minImprovement < lastLogp)
		{   //***maybe put some variable here to define what is minimal improvement??
			if (alphaDivide == 0)
				alphaDivide = 1;
			else
				break;
		}

		// if (alphaDivide) alpha/=2;修改学习率
		if (alphaDivide) alpha/=alphaDiv;//修改学习率

		lastLogp = logp;
	}

	// final entropy and perplexity on test data
	// this is for the test data
	logp = 0;
	wordCounter = 0;
	flushNet();
	testNetFile(testFile);
	printf("final TEST entropy: %.4f (%.4f)\n", -logp/log10(2)/wordCounter, exp10(-logp/(double)wordCounter));

	//////////////////////////////////////////////////////////////////////////////////////////////
	if(saveModel > 0)
	{
		cout << "saving final model!" << endl;
		char modelName[1024];
		sprintf(modelName, "%s_%f.model", modelFile, oriAlpha);
		saveNet(modelName);
		cout << "saving final model done!" << endl;
	}
}

// outConditionDSyn still missing
void RNNPG::saveSynapse(FILE *fout)
{
	int i, j, N;
	for(i = 0; i < MAX_CON_N; i ++)
	{
		fprintf(fout, "convolutional matrix %d:\n", i);
		N = hiddenSize * conSeq[i];
		for(j = 0; j < N; j ++)
			fprintf(fout, "%.16g\n", conSyn[i][j].weight);
		fprintf(fout, "\n");
	}
	fprintf(fout, "\n\n");

	fprintf(fout, "word embedding matrix:\n");
	N = vocab.getVocabSize() * hiddenSize;
	for(i = 0; i < N; i ++)
		fprintf(fout, "%.16g\n", senweSyn[i].weight);
	fprintf(fout, "\n\n");

	N = hiddenSize * 2 * hiddenSize;
	fprintf(fout, "compress matrix:\n");
	for(i = 0; i < N; i ++)
		fprintf(fout, "%.16g\n", compressSyn[i].weight);
	fprintf(fout, "\n\n");

	for(i = 0; i < 8; i ++)
	{
		fprintf(fout, "7 character map matrix %d:\n", i);
		N = hiddenSize * hiddenSize;
		for(j = 0; j < N; j ++)
			fprintf(fout, "%.16g\n", map7Syn[i][j].weight);
		fprintf(fout, "\n");
	}
	for(i = 0; i < 6; i ++)
	{
		fprintf(fout, "5 character map matrix %d:\n", i);
		N = hiddenSize * hiddenSize;
		for(j = 0; j < N; j ++)
			fprintf(fout, "%.16g\n", map5Syn[i][j].weight);
		fprintf(fout, "\n");
	}
	fprintf(fout, "\n\n");

	fprintf(fout, "weight matrix from hidden layer to input layer:\n");
	N = hiddenSize * (hiddenSize + hiddenSize + vocab.getVocabSize());
	for(i = 0; i < N; i ++)
		fprintf(fout, "%.16g\n", hiddenInSyn[i].weight);
	fprintf(fout, "\n\n");

	fprintf(fout, "weight matrix from output layer to hidden layer:\n");
	N = (vocab.getVocabSize() + classSize) * hiddenSize;
	for(i = 0; i < N; i ++)
		fprintf(fout, "%.16g\n", outHiddenSyn[i].weight);
	fprintf(fout, "\n\n");

	fprintf(fout, "direct error matrix from input layer to output layer:\n");
	N = (vocab.getVocabSize()+classSize) * hiddenSize;
	for(i = 0; i < N; i ++)
		fprintf(fout, "%.16g\n", outConditionDSyn[i].weight);
	fprintf(fout, "\n\n");
}

// outConditionDSyn still missing
void RNNPG::loadSynapse(FILE *fin)
{
	int i, j, N;
	for(i = 0; i < MAX_CON_N; i ++)
	{
		skiputil(':', fin);
		N = hiddenSize * conSeq[i];
		for(j = 0; j < N; j ++)
			fscanf(fin, "%lf\n", &conSyn[i][j].weight);
	}

	skiputil(':', fin);
	N = vocab.getVocabSize() * hiddenSize;
	for(i = 0; i < N; i ++)
		fscanf(fin, "%lf", &senweSyn[i].weight);

	skiputil(':', fin);
	N = hiddenSize * 2 * hiddenSize;
	for(i = 0; i < N; i ++)
		fscanf(fin, "%lf", &compressSyn[i].weight);

	for(i = 0; i < 8; i ++)
	{
		skiputil(':', fin);
		N = hiddenSize * hiddenSize;
		for(j = 0; j < N; j ++)
			fscanf(fin, "%lf", &map7Syn[i][j].weight);
	}
	for(i = 0; i < 6; i ++)
	{
		skiputil(':', fin);
		N = hiddenSize * hiddenSize;
		for(j = 0; j < N; j ++)
			fscanf(fin, "%lf", &map5Syn[i][j].weight);
	}
	skiputil(':', fin);
	N = hiddenSize * (vocab.getVocabSize() + hiddenSize + hiddenSize);
	for(i = 0; i < N; i ++)
		fscanf(fin, "%lf", &hiddenInSyn[i].weight);

	skiputil(':', fin);
	N = (vocab.getVocabSize() + classSize) * hiddenSize;
	for(i = 0; i < N; i ++)
		fscanf(fin, "%lf", &outHiddenSyn[i].weight);

	skiputil(':', fin);
	N = (vocab.getVocabSize()+classSize) * hiddenSize;
	for(i = 0; i < N; i ++)
		fscanf(fin, "%lf", &outConditionDSyn[i].weight);
}

void RNNPG::loadNet(const char *infile)
{
	FILE *fin = xfopen(infile, "rb");
	loadBasicSetting(fin);
	vocab.load(fin);
	if(inNeu == NULL)
		initNet();
	loadSynapse(fin);
	loadNeuron(fin);
	fclose(fin);
}

void RNNPG::saveNet(const char *outfile)
{
	FILE *fout = xfopen(outfile, "wb");
	saveBasicSetting(fout);
	vocab.save(fout);
	saveSynapse(fout);
	saveNeuron(fout);
	fclose(fout);
}

void RNNPG::saveNeuron(FILE *fout)
{
	int i, j, N, unitNum;
	unitNum = 7;
	for(i = 0; i < SEN7_HIGHT; i ++)
	{
		N = unitNum * hiddenSize;
		fprintf(fout, "neuron in 7 char sentence model, layer %d:\n", i);
		for(j = 0; j < N; j ++)
			fprintf(fout, "%.16g %.16g\n", sen7Neu[i][j].ac, sen7Neu[i][j].er);
		fprintf(fout, "\n");
		if(unitNum > 1)	unitNum -= conSeq[i] - 1;
	}
	fprintf(fout, "\n\n");

	unitNum = 5;
	for(i = 0; i < SEN5_HIGHT; i ++)
	{
		N = unitNum * hiddenSize;
		fprintf(fout, "neuron in 5 char sentence model, layer %d:\n", i);
		for(j = 0; j < N; j ++)
			fprintf(fout, "%.16g %.16g\n", sen5Neu[i][j].ac, sen5Neu[i][j].er);
		fprintf(fout, "\n");
		if(unitNum > 1)	unitNum -= conSeq[i] - 1;
	}
	fprintf(fout, "\n\n");

	fprintf(fout, "history neuron:\n");
	N = hiddenSize;
	for(i = 0; i < N; i ++)
		fprintf(fout, "%.16g %.16g\n", hisNeu[i].ac, hisNeu[i].er);
	fprintf(fout, "\n\n");

	fprintf(fout, "combine neuron:\n");
	N = hiddenSize * 2;
	for(i = 0; i < N; i ++)
		fprintf(fout, "%.16g %.16g\n", cmbNeu[i].ac, cmbNeu[i].er);
	fprintf(fout, "\n\n");

	fprintf(fout, "condition neuron:\n");
	N = hiddenSize;
	for(i = 0; i < N; i ++)
		fprintf(fout, "%.16g %.16g\n", conditionNeu[i].ac, conditionNeu[i].er);
	fprintf(fout, "\n\n");

	fprintf(fout, "input neuron:\n");
	N = vocab.getVocabSize() + hiddenSize + hiddenSize;
	for(i = 0; i < N; i ++)
		fprintf(fout, "%.16g %.16g\n", inNeu[i].ac, inNeu[i].er);
	fprintf(fout, "\n\n");

	fprintf(fout, "hidden neuron:\n");
	N = hiddenSize;
	for(i = 0; i < N; i ++)
		fprintf(fout, "%.16g %.16g\n", hiddenNeu[i].ac, hiddenNeu[i].er);
	fprintf(fout, "\n\n");

	fprintf(fout, "output neuron:\n");
	N = vocab.getVocabSize() + classSize;
	for(i = 0; i < N; i ++)
		fprintf(fout, "%.16g %.16g\n", outNeu[i].ac, outNeu[i].er);
	fprintf(fout, "\n\n");
}

void RNNPG::loadNeuron(FILE *fin)
{
	int i, j, N, unitNum;
	unitNum = 7;
	for(i = 0; i < SEN7_HIGHT; i ++)
	{
		N = unitNum * hiddenSize;
		skiputil(':', fin);
		for(j = 0; j < N; j ++)
			fscanf(fin, "%lf %lf", &sen7Neu[i][j].ac, &sen7Neu[i][j].er);
		if(unitNum > 1)	unitNum -= conSeq[i] - 1;
	}

	unitNum = 5;
	for(i = 0; i < SEN5_HIGHT; i ++)
	{
		N = unitNum * hiddenSize;
		skiputil(':', fin);
		for(j = 0; j < N; j ++)
			fscanf(fin, "%lf %lf", &sen5Neu[i][j].ac, &sen5Neu[i][j].er);
		if(unitNum > 1)	unitNum -= conSeq[i] - 1;
	}

	skiputil(':', fin);
	N = hiddenSize;
	for(i = 0; i < N; i ++)
		fscanf(fin, "%lf %lf", &hisNeu[i].ac, &hisNeu[i].er);

	skiputil(':', fin);
	N = hiddenSize * 2;
	for(i = 0; i < N; i ++)
		fscanf(fin, "%lf %lf", &cmbNeu[i].ac, &cmbNeu[i].er);

	skiputil(':', fin);
	N = hiddenSize;
	for(i = 0; i < N; i ++)
		fscanf(fin, "%lf %lf", &conditionNeu[i].ac, &conditionNeu[i].er);

	skiputil(':', fin);
	N = vocab.getVocabSize() + hiddenSize + hiddenSize;
	for(i = 0; i < N; i ++)
		fscanf(fin, "%lf %lf", &inNeu[i].ac, &inNeu[i].er);

	skiputil(':', fin);
	N = hiddenSize;
	for(i = 0; i < N; i ++)
		fscanf(fin, "%lf %lf", &hiddenNeu[i].ac, &hiddenNeu[i].er);

	skiputil(':', fin);
	N = vocab.getVocabSize() + classSize;
	for(i = 0; i < N; i ++)
		fscanf(fin, "%lf %lf", &outNeu[i].ac, &outNeu[i].er);
}

void RNNPG::saveBasicSetting(FILE *fout)
{
	fprintf(fout, "alpha:%.16g\n", alpha);
	fprintf(fout, "beta:%.16g\n", beta);
	fprintf(fout, "hiddenSize:%d\n", hiddenSize);
	fprintf(fout, "classSize:%d\n", classSize);
	fprintf(fout, "max iter:%d\n", maxIter);
	fprintf(fout, "fix first:%d\n", fixSentenceModelFirstLayer);
	fprintf(fout, "rand init:%d\n", randomlyInitSenModelEmbedding);
	fprintf(fout, "train file:%s\n", trainFile);
	fprintf(fout, "valid file:%s\n", validFile);
	fprintf(fout, "test file:%s\n", testFile);
	fprintf(fout, "word embeding file:%s\n", wordEmbeddingFile);
	fprintf(fout, "random seed:%d\n", randomSeed);
	fprintf(fout, "min Improvement:%.16g\n", minImprovement);
	// fprintf(fout, "\n\n");

	fprintf(fout, "word counter:%d\n", wordCounter);
	fprintf(fout, "total poem count:%d\n", totalPoemCount);
	fprintf(fout, "log prob:%.16g\n", logp);
	fprintf(fout, "alpha divide:%d\n", alphaDivide);
	fprintf(fout, "stableAC:%.16g\n", stableAC);
	fprintf(fout, "flush option:%d\n", flushOption);
	fprintf(fout, "minimum value for convolutional matrix:%.16g\n", consynMin);
	fprintf(fout, "maximum value for convolutional matrix:%.16g\n", consynMax);
	fprintf(fout, "offset of the value for convolutional matrix:%.16g\n", consynOffset);
	fprintf(fout, "the direct error from output layer to condition layer:%d\n", directError);
	fprintf(fout, "BPTT training for the recurrent context model (during learning the sentence model):%d\n", conbptt);
	fprintf(fout, "\n\n");
}

void RNNPG::loadBasicSetting(FILE *fin)
{
	skiputil(':', fin);	fscanf(fin, "%lf", &alpha);
	skiputil(':', fin);	fscanf(fin, "%lf", &beta);
	skiputil(':', fin);	fscanf(fin, "%d", &hiddenSize);
	skiputil(':', fin);	fscanf(fin, "%d", &classSize);
	skiputil(':', fin);	fscanf(fin, "%d", &maxIter);
	int bval;
	skiputil(':', fin);	fscanf(fin, "%d", &bval);
	fixSentenceModelFirstLayer = bval;
	skiputil(':', fin);	fscanf(fin, "%d", &bval);
	randomlyInitSenModelEmbedding = bval;
	skiputil(':', fin);	fscanf(fin, "%s", trainFile);
	skiputil(':', fin);	fscanf(fin, "%s", validFile);
	skiputil(':', fin);	fscanf(fin, "%s", testFile);
	skiputil(':', fin);	fscanf(fin, "%s", wordEmbeddingFile);
	skiputil(':', fin);	fscanf(fin, "%d", &randomSeed);
	skiputil(':', fin);	fscanf(fin, "%lf", &minImprovement);
	skiputil(':', fin);	fscanf(fin, "%d", &wordCounter);
	skiputil(':', fin);	fscanf(fin, "%d", &totalPoemCount);
	skiputil(':', fin);	fscanf(fin, "%lf", &logp);
	skiputil(':', fin);	fscanf(fin, "%d", &alphaDivide);
	skiputil(':', fin);	fscanf(fin, "%lf", &stableAC);
	skiputil(':', fin);	fscanf(fin, "%d", &flushOption);
	skiputil(':', fin);	fscanf(fin, "%lf", &consynMin);
	skiputil(':', fin);	fscanf(fin, "%lf", &consynMax);
	skiputil(':', fin);	fscanf(fin, "%lf", &consynOffset);
	skiputil(':', fin);	fscanf(fin, "%d", &bval);
	directError = bval;
	skiputil(':', fin);	fscanf(fin, "%d", &bval);
	conbptt = bval;
}

void RNNPG::showParameters()
{
	// printf("-conf <conf file>  -- configuration file, the program first read the configuration file options and then use the command line options. That is to say command line options have higher priorities\n");
	printf("-alpha %f\n", alpha);
	printf("-alphaDiv %f\n", alphaDiv);
	printf("-beta  %g\n", beta);
	printf("-hidden %d      -- hidden layer size\n", hiddenSize);
	printf("-class  %d      -- class size\n", classSize);
	printf("-iter   %d      -- max iteration\n", maxIter);
	printf("-fixFirst %s   -- true means fix the first layer of the sentence model\n", fixSentenceModelFirstLayer ? "true" : "false");
	printf("-randInit %s   -- true means randomly initialize sentence model word embedding; false means using word2vec to initialize the layer\n",
			randomlyInitSenModelEmbedding ? "true" : "false");
	printf("-trainF    %s\n", trainFile);
	printf("-validF    %s\n", validFile);
	printf("-testF     %s\n", testFile);
	printf("-embedF    %s\n", wordEmbeddingFile);
	printf("-saveModel %d  -- 0 do not save model; 1 save every 10000 poems; 2 save at the end\n", saveModel);
	printf("-modelF    %s\n", modelFile);
	printf("-randSeed  %d   -- random seed for all the random initialization\n", randomSeed);
	printf("-minImp  %f   -- minmum improvment for convergence, default 1.0001\n", minImprovement);
	printf("-stableAC %f  -- everytime when flushing the network, activation in last time hidden layer is set to 'stableAC', default 0.1\n", stableAC);
	printf("-flushOption %d  -- 1, flush the network every sentence; 2, flush the network every poem; 3, never flush (only flush at the beginning of training)\n", flushOption);
	printf("-consynMin %f  -- minimum value for convolutional matrix\n", consynMin);
	printf("-consynMax %f  -- maximum value for convolutional matrix\n", consynMax);
	printf("-consynOffset %f -- offset of the value for convolutional matrix\n", consynOffset);
	printf("-direct %s -- true means the direct error from output layer to condition layer will be used\n",
			directError ? "true" : "false");
	printf("-conbptt %s -- true means use BPTT training for the recurrent context model (during learning the sentence model)\n",
				conbptt ? "true" : "false");
	printf("-vocabClassF %s\n", vocabClassF);
	printf("-perSentUpdate %s\n", perSentUpdate ? "true" : "false");
	printf("-historyStableAC %f -- stableAC in recurrent context model\n", historyStableAC);
	printf("-adaGrad %s -- true means using AdaGrad training algorithm; false means using SDG\n", adaGrad ? "true" : "false");
}

// this function is from Tomas Mikolov's toolkit, rnnlm-0.2b
/**
 * @brief
 * 实现矩阵和向量的乘法，注意，如果目标向量里已经有值，这个方法是不会将目标向量里原来的的值清空掉的，而是直接在原有的值的基础上加上新计算出来的值！！！所以要实现覆盖的模式，必须调用clearNeurons来清空
 * @param dest 目标向量
 * @param srcvec 源向量
 * @param srcmatrix 源矩阵
 * @param matrix_width 矩阵的列数
 * @param from 正向传播中目标向量的开始维度，反向传播中源向量的开始维度
 * @param to 正向传播中目标向量的结束维度，反向传播中源向量的结束维度
 * @param from2 正向传播中源向量的开始维度，反向传播中目标向量的开始维度
 * @param to2 正向传播中源向量的结束维度，反向传播中目标向量的结束维度
 * @param type 选择乘法的类型，0代表进行前向传播的乘法，1代表进行反向传播的乘法，在反向传播的过程中自带转置效果
 */
void RNNPG::matrixXvector(struct neuron *dest, struct neuron *srcvec, struct synapse *srcmatrix, int matrix_width, int from, int to, int from2, int to2, int type)
{
    int a, b;
    double val1, val2, val3, val4;
    double val5, val6, val7, val8;

    if (type==0) {		//ac mod
	for (b=0; b<(to-from)/8; b++) {
	    val1=0;
	    val2=0;
	    val3=0;
	    val4=0;

	    val5=0;
	    val6=0;
	    val7=0;
	    val8=0;

	    for (a=from2; a<to2; a++) {
    		val1 += srcvec[a].ac * srcmatrix[a+(b*8+from+0)*matrix_width].weight;
    		val2 += srcvec[a].ac * srcmatrix[a+(b*8+from+1)*matrix_width].weight;
    		val3 += srcvec[a].ac * srcmatrix[a+(b*8+from+2)*matrix_width].weight;
    		val4 += srcvec[a].ac * srcmatrix[a+(b*8+from+3)*matrix_width].weight;

    		val5 += srcvec[a].ac * srcmatrix[a+(b*8+from+4)*matrix_width].weight;
    		val6 += srcvec[a].ac * srcmatrix[a+(b*8+from+5)*matrix_width].weight;
    		val7 += srcvec[a].ac * srcmatrix[a+(b*8+from+6)*matrix_width].weight;
    		val8 += srcvec[a].ac * srcmatrix[a+(b*8+from+7)*matrix_width].weight;
    	    }
    	    dest[b*8+from+0].ac += val1;
    	    dest[b*8+from+1].ac += val2;
    	    dest[b*8+from+2].ac += val3;
    	    dest[b*8+from+3].ac += val4;

    	    dest[b*8+from+4].ac += val5;
    	    dest[b*8+from+5].ac += val6;
    	    dest[b*8+from+6].ac += val7;
    	    dest[b*8+from+7].ac += val8;
	}

	for (b=b*8; b<to-from; b++) {
	    for (a=from2; a<to2; a++) {
    		dest[b+from].ac += srcvec[a].ac * srcmatrix[a+(b+from)*matrix_width].weight;
    	    }
    	}
    }
    else {		//er mod
    	for (a=0; a<(to2-from2)/8; a++) {
	    val1=0;
	    val2=0;
	    val3=0;
	    val4=0;

	    val5=0;
	    val6=0;
	    val7=0;
	    val8=0;

	    for (b=from; b<to; b++) {
    	        val1 += srcvec[b].er * srcmatrix[a*8+from2+0+b*matrix_width].weight;
    	        val2 += srcvec[b].er * srcmatrix[a*8+from2+1+b*matrix_width].weight;
    	        val3 += srcvec[b].er * srcmatrix[a*8+from2+2+b*matrix_width].weight;
    	        val4 += srcvec[b].er * srcmatrix[a*8+from2+3+b*matrix_width].weight;

    	        val5 += srcvec[b].er * srcmatrix[a*8+from2+4+b*matrix_width].weight;
    	        val6 += srcvec[b].er * srcmatrix[a*8+from2+5+b*matrix_width].weight;
    	        val7 += srcvec[b].er * srcmatrix[a*8+from2+6+b*matrix_width].weight;
    	        val8 += srcvec[b].er * srcmatrix[a*8+from2+7+b*matrix_width].weight;
    	    }
    	    dest[a*8+from2+0].er += val1;
    	    dest[a*8+from2+1].er += val2;
    	    dest[a*8+from2+2].er += val3;
    	    dest[a*8+from2+3].er += val4;

    	    dest[a*8+from2+4].er += val5;
    	    dest[a*8+from2+5].er += val6;
    	    dest[a*8+from2+6].er += val7;
    	    dest[a*8+from2+7].er += val8;
	}

	for (a=a*8; a<to2-from2; a++) {
	    for (b=from; b<to; b++) {
    		dest[a+from2].er += srcvec[b].er * srcmatrix[a+from2+b*matrix_width].weight;
    	    }
    	}

		//控制误差的范围，避免梯度爆炸
    	for (a=from2; a<to2; a++) {
    	    if (dest[a].er>15) dest[a].er=15;
    	    if (dest[a].er<-15) dest[a].er=-15;
    	}
    }
}

/**
 * Since given all the previous sentences, the resulting activation caused by these sentences in hiddenNeu is constant,
 * we pre-compute all the activations here
 */
void RNNPG::getContextHiddenNeu(vector<string> &prevSents, neuron **contextHiddenNeu)
{
	int i, SEN_HIGHT = -1;
	neuron **senNeu = NULL;
	synapse **mapSyn = NULL;
	vector<string> words;
	clearNeurons(cmbNeu, hiddenSize * 2, 3);

	if(flushOption == EVERY_POEM)
	{
		flushNet();
		for(i = 0; i < hiddenSize; i ++)
		{
			hiddenNeu[i].ac = stableAC;
			hiddenNeu[i].er = 0;
		}
	}

	// obtaining hisNeu from all previous sentences
	for(i = 0; i < (int)prevSents.size(); i ++)
	{
		words.clear();
		split(prevSents[i], " ", words);
		SEN_HIGHT = words.size() == 5 ? SEN5_HIGHT : SEN7_HIGHT;
		senNeu = words.size() == 5 ? sen5Neu : sen7Neu;
		initSent(words.size());
		neuron *sen_repr = sen2vec(words, senNeu, SEN_HIGHT);		// this is the pointer for the top layer sentence model, DO NOT modify it
		memcpy(cmbNeu + hiddenSize, sen_repr, sizeof(neuron)*hiddenSize);
		clearNeurons(hisNeu, hiddenSize, 3);
		matrixXvector(hisNeu, cmbNeu, compressSyn, hiddenSize * 2, 0, hiddenSize, 0, hiddenSize * 2, 0);
		funACNeurons(hisNeu, hiddenSize);

		if(flushOption == EVERY_POEM && i + 1 < (int)prevSents.size())
		{
			if(mapSyn == NULL)
				mapSyn = words.size() == 5 ? map5Syn : map7Syn;
			words.clear();
			split(prevSents[i + 1], " ", words);
			words.push_back("</s>");	// during generation, we DO care about the End-of-Sentence
			int lastWord = 0, curWord = -1, wdPos;
			for(wdPos = 0; wdPos < (int)words.size(); wdPos ++)
			{
				curWord = vocab.getVocabID(words[wdPos].c_str());
				bool isRare = false;
				if(curWord == -1)
				{
					// when the word cannot be found, we use <R> instead
					curWord = vocab.getVocabID("<R>");
					isRare = true;
				}
				assert(curWord != -1);		// this is impossible, or there is a bug!
				inNeu[lastWord].ac = 1;
				computeNet(lastWord, curWord, wdPos, mapSyn);
				inNeu[lastWord].ac = 0;
				copyHiddenLayerToInput();
				lastWord = curWord;
			}
			words.pop_back();	// generation done and delete </s>
		}

		if(i != (int)prevSents.size() - 1)
			memcpy(cmbNeu, hisNeu, sizeof(neuron)*hiddenSize);
	}

	// after obtaining hisNeu, we will next get the conditionNeu and activation from conditionNeu to hiddenNeu
	int SEN_LEN = words.size() + 1;
	mapSyn = words.size() == 5 ? map5Syn : map7Syn;
	int V = vocab.getVocabSize();
	for(i = 0; i < SEN_LEN; i ++)
	{
		clearNeurons(inNeu + V, hiddenSize, 1);
		matrixXvector(inNeu + V, hisNeu, mapSyn[i], hiddenSize, 0, hiddenSize, 0, hiddenSize, 0);
		funACNeurons(inNeu + V, hiddenSize);
		clearNeurons(contextHiddenNeu[i], hiddenSize, 1);
		matrixXvector(contextHiddenNeu[i], inNeu, hiddenInSyn, V + hiddenSize + hiddenSize, 0, hiddenSize, V, V + hiddenSize, 0);

		// printNeurons(contextHiddenNeu[i], hiddenSize);
	}
}

/**
 * @brief
 * 比较两个pair<string,double>，如果p1.second<p2.second，返回0，否则返回1
 * @param p1 pair<string,double>的引用
 * @param p2 pair<string,double>的引用
 * @return bool 如果p1.second<p2.second，返回0，否则返回1
 */
bool pair_cmp(const pair<string,double> &p1, const pair<string,double> &p2)
{
	return !(p1.second < p2.second);
}

void RNNPG::computeNetContext(const char *lword, const char *cword, neuron *curHiddenNeu, neuron *contextHiddenNeu,
		neuron *newHiddenNeu, vector<pair<string,double> > &nextWordProbs)
{
	int lastWord = vocab.getVocabID(lword);
	int i, j, V = vocab.getVocabSize();

	// recurrent from last time hidden layer: put last time hidden layer in the input layer
	if(lastWord == 0)	// if lword == "</s>"
	{
		if(flushOption == EVERY_SENTENCE)
			flushNet();
		else if(flushOption == EVERY_POEM)
			copyNeurons(inNeu + V + hiddenSize, curHiddenNeu, hiddenSize, 1);
		else
		{
			fprintf(stderr, "flushOption %d not supported right now in computeNetContext (for decoding)\n", flushOption);
			exit(1);
		}
	}
	else
		copyNeurons(inNeu + V + hiddenSize, curHiddenNeu, hiddenSize, 1);

	copyNeurons(hiddenNeu, contextHiddenNeu, hiddenSize, 1); // if lword == 0, the copy is invalid, since flushNet will take care of it  BUG!!!!!!! Now fixed by change the sequence the lines of code above


	int N = V + hiddenSize + hiddenSize;
	matrixXvector(hiddenNeu, inNeu, hiddenInSyn, N, 0, hiddenSize, V + hiddenSize, V + hiddenSize + hiddenSize, 0);
	for(i = 0; i < hiddenSize; i ++)
		hiddenNeu[i].ac += hiddenInSyn[i*N + lastWord].weight;
	funACNeurons(hiddenNeu, hiddenSize);

	copyNeurons(newHiddenNeu, hiddenNeu, hiddenSize, 1);

	// compute probabilities on classes
	clearNeurons(outNeu + V, classSize, 1);
	matrixXvector(outNeu, hiddenNeu, outHiddenSyn, hiddenSize, V, V + classSize, 0, hiddenSize, 0);
	double sum = 0;
	N = V + classSize;
	for(i = V; i < N; i ++)
	{
		if(outNeu[i].ac < -50) outNeu[i].ac = -50;
		if(outNeu[i].ac > 50) outNeu[i].ac = 50;
		outNeu[i].ac = FAST_EXP(outNeu[i].ac);
		sum += outNeu[i].ac;
	}
	for(i = V; i < N; i ++)
		outNeu[i].ac /= sum;

	// vector<pair<string,double>>
	nextWordProbs.clear();
	if(cword == NULL)
	{
		// oh my god, the computational cost is great!!!
		matrixXvector(outNeu, hiddenNeu, outHiddenSyn, hiddenSize, 0, V, 0, hiddenSize, 0);
		for(i = 0; i < classSize; i ++)
		{
			sum = 0;
			// cout << "class start " << classStart[i] << ", class end " << classEnd[i] << endl;
			for(j = classStart[i]; j < classEnd[i]; j ++)
			{
				if(outNeu[j].ac < -50) outNeu[j].ac = -50;
				if(outNeu[j].ac > 50) outNeu[j].ac = 50;
				outNeu[j].ac = FAST_EXP(outNeu[j].ac);
				sum += outNeu[j].ac;
			}
			for(j = classStart[i]; j < classEnd[i]; j ++)
			{
				outNeu[j].ac /= sum;
				nextWordProbs.push_back(make_pair(this->voc_arr[j].wd, outNeu[V+i].ac * outNeu[j].ac));
			}
		}
		sort(nextWordProbs.begin(), nextWordProbs.end(), pair_cmp);
	}
	else
	{
		int curWord = vocab.getVocabID(cword);
		int classIndex = voc_arr[curWord].classIndex;
		matrixXvector(outNeu, hiddenNeu, outHiddenSyn, hiddenSize, classStart[classIndex], classEnd[classIndex], 0, hiddenSize, 0);
		sum = 0;
		for(j = classStart[classIndex]; j < classEnd[classIndex]; j ++)
		{
			if(outNeu[j].ac < -50) outNeu[j].ac = -50;
			if(outNeu[j].ac > 50) outNeu[j].ac = 50;
			outNeu[j].ac = FAST_EXP(outNeu[j].ac);
			sum += outNeu[j].ac;
		}
		for(j = classStart[classIndex]; j < classEnd[classIndex]; j ++)
			outNeu[j].ac /= sum;
		nextWordProbs.push_back(make_pair(this->voc_arr[curWord].wd, outNeu[V+classIndex].ac * outNeu[curWord].ac));
	}
}

double RNNPG::computeNetContext(const char *lword, int startPos, const vector<string> &words, neuron *curHiddenNeu, neuron **contextHiddenNeus,
			neuron *newHiddenNeu)
{
	double phraseLogProb = 0;
	int lastWord = vocab.getVocabID(lword);
	if(lastWord == -1) lastWord = vocab.getVocabID("<R>");
	int i, j, V = vocab.getVocabSize();

	// recurrent from last time hidden layer: put last time hidden layer in the input layer
	if(lastWord == 0)	// if lword == "</s>"
	{
		if(flushOption == EVERY_SENTENCE)
			flushNet();
		else if(flushOption == EVERY_POEM)
			copyNeurons(inNeu + V + hiddenSize, curHiddenNeu, hiddenSize, 1);
		else
		{
			fprintf(stderr, "flushOption %d not supported right now in computeNetContext (for decoding)\n", flushOption);
			exit(1);
		}
	}
	else
		copyNeurons(inNeu + V + hiddenSize, curHiddenNeu, hiddenSize, 1);

	// cout << "copy activation done!" << endl;

	int curPos = startPos, N = V + hiddenSize + hiddenSize;
	for(size_t idx = 0; idx < words.size(); idx ++)
	{
		// cout << "word = " << words[idx] << endl;

		copyNeurons(hiddenNeu, contextHiddenNeus[curPos], hiddenSize, 1);
		matrixXvector(hiddenNeu, inNeu, hiddenInSyn, N, 0, hiddenSize, V + hiddenSize, V + hiddenSize + hiddenSize, 0);
		for(i = 0; i < hiddenSize; i ++)
			hiddenNeu[i].ac += hiddenInSyn[i*N + lastWord].weight;
		funACNeurons(hiddenNeu, hiddenSize);

		// cout << "compute activation done" << endl;

		// compute probs on classes
		clearNeurons(outNeu + V, classSize, 1);
		matrixXvector(outNeu, hiddenNeu, outHiddenSyn, hiddenSize, V, V + classSize, 0, hiddenSize, 0);
		double sum = 0;
		N = V + classSize;
		for(i = V; i < N; i ++)
		{
			if(outNeu[i].ac < -50) outNeu[i].ac = -50;
			if(outNeu[i].ac > 50) outNeu[i].ac = 50;
			outNeu[i].ac = FAST_EXP(outNeu[i].ac);
			sum += outNeu[i].ac;
		}
		for(i = V; i < N; i ++)
			outNeu[i].ac /= sum;

		// cout << "on classes done" << endl;

		// compute probs on words in the current class
		int curWord = vocab.getVocabID(words[idx].c_str());

		// cout << "curWord = " << curWord << endl;

		if(curWord == -1)
			curWord = vocab.getVocabID("<R>");

		int classIndex = voc_arr[curWord].classIndex;
		matrixXvector(outNeu, hiddenNeu, outHiddenSyn, hiddenSize, classStart[classIndex], classEnd[classIndex], 0, hiddenSize, 0);
		sum = 0;
		for(j = classStart[classIndex]; j < classEnd[classIndex]; j ++)
		{
			if(outNeu[j].ac < -50) outNeu[j].ac = -50;
			if(outNeu[j].ac > 50) outNeu[j].ac = 50;
			outNeu[j].ac = FAST_EXP(outNeu[j].ac);
			sum += outNeu[j].ac;
		}
		for(j = classStart[classIndex]; j < classEnd[classIndex]; j ++)
			outNeu[j].ac /= sum;

		phraseLogProb += log(outNeu[V+classIndex].ac * outNeu[curWord].ac);
		lastWord = curWord;

		curPos ++;
		if(idx != words.size() - 1)
			copyHiddenLayerToInput();
		else
			copyNeurons(newHiddenNeu, hiddenNeu, hiddenSize, 1);
	}

	// cout << "all the computation done " << endl;

	return phraseLogProb;
}

/**
 * @brief
 * 测试网络
 */
void RNNPG::testNet()
{
	mode = TEST_MODE;

	cout << "test net" << endl;
	cout << "modelF = " << modelFile << endl;
	cout << "testF = " << testFile << endl;

	showParameters();

	loadNet(modelFile);
	logp = 0;
	wordCounter = 0;
	flushNet();
	testNetFile(testFile);
	printf("final TEST entropy: %.4f (%.4f)\n", -logp/log10(2)/wordCounter, exp10(-logp/(double)wordCounter));
}

