/*
 * main.cpp
 *
 *  Created on: 27 Dec 2013
 *      Author: s1270921
 */

/* *
 * ================= change log ======================
 * 0. Today Sun 16 Feb 2014, I began to write change log in every version
 *
 * 1. model saving is problematic: sprintf(modelName, "%s_%f.model", modelFile, alpha); // The alpha is the final alpha
 * and therefore cannot be used to distinguish different models. Add oriAlpha
 *
 * 2. 23 Feb, 2014. Add new classes to vocabulary. The classes are obtained by clustering the word2vec output. word2vec trained on all poem corpus
 * and we use cluto (./vcluster -clmethod=bagglo -crfun=i2 $clinF 100) to get the resulting classes
 * 		2.1 Vocab.h add void loadVocabClass(const char *infile) and void getClassStartEnd(int *classStart, int *classEnd, int classSize) to support
 * 			new kind of vocab class
 * 		2.2 change sort(vocab, vocab + vocabSize) to qsort. It seems that use sort will course problem... But will only when classSize == 50
 *
 * 3. saving beta might be 0, in saveBasicSetting and showParameters, use %g instead of %f
 *
 * 4. 24 Feb, 2014. Add per sentence update during BPTT training
 *
 * 5. Add historyStableAC to the recurrent context model, the value 0.01 is currently the best. This value doesn't influence the results much
 *
 * 6. add adaGrad flag to control if we use the adaptive gradient descent training algorithm. Initial experiment does not show any improvement.
 * 		Now I will turn off this option
 *
 * 7. save model with more precision. %.16g
 *
 * ------------- Decoder -------------
 * 1. support flush option 2 (flush the neuron per poem rather than per sentence)
 * 2. add translation table this time P(S_i|F_i)_max = max{ P_{rnnpg}(S_i) * P(F_i|S_i)}
 * 			S_i is the second sentence and F_i is the first sentence
 * 		2.1 - Delete TransTable.h and TransTable.cpp as this is not actually a translation table
 * 		2.2 - Add a new TranslationTable.h StringBuffer.h StringBuffer.cpp utf8.h utf8.cpp (discarded)
 * 		2.3 - int decodeTransTable(vector<string> &prevSents, int stackSize, int K, vector<string> &topSents); and int decodeTransTable(const char* infile, const char* outfile, int stackSize, int K, int startId = 0);
 * 			to support translation Table
 * 		2.4 - add options (channelOption) for selecting different channel model features
 * 		2.5 - add a re-ranking strategy in the final stack with the rnnpg score
 * 		2.6 - add disableRNN to disable RNN feature and see what happened
 * 		2.7 - support MERT training, add -weightFile add feature weight file to the decoder
 */

#include <iostream>
#include <cstdio>
using namespace std;

#include "RNNPG.h"
#include "xutil.h"
#include "Config.h"
#include "Decoder.h"

/**
 * @brief
 * 这个函数的功能是给定返回argv这个字符串指针中字符串curArg的数组下标，argc给定数组的长度
 * @param argc
 * @param argv
 * @param curArg
 * @return int -1：没有找到指定的参数;other：返回下标的index
 */
int getArgPos(int argc, char **argv, const char *curArg)
{
	for(int i = 1; i < argc; i ++)
		if(strcmp(argv[i], curArg) == 0)
			return i;
	return -1;
}

/*
 * 这是rnnpg程序运行的一些参数
-conf <conf file>  -- configuration file, the program first read the configuration file options and then use the command line options. That is to say command line options have higher priorities
-alpha <double>    -- learning rate
-beta  <double>    -- regulation weight
-hidden <int>      -- hidden layer size
-class  <int>      -- class size
-iter   <int>      -- max iteration
-fixFirst <bool>   -- true means fix the first layer of the sentence model
-randInit <bool>   -- true means randomly initialize sentence model word embedding; false means using word2vec to initialize the layer
-trainF    <train file>
-validF    <valid file>
-testF     <test file>
-embedF    <word embeding file>
-saveModel <bool>  -- true means model during training
-modelF    <model file>
-randSeed  <int>    -- random seed for all the random initialization
-minImp  <double>   -- minmum improvment for convergence, default 1.0001
-stableAC <double>  -- everytime when flushing the network, activation in last time hidden layer is set to 'stableAC', default 0.1
-flushOption <int>  -- 1, flush the network every sentence; 2, flush the network every poem; 3, never flush (only flush at the beginning of training)
-consynMin <float>  -- minimum value for convolutional matrix
-consynMax <float>  -- maximum value for convolutional matrix
-consynOffset <float> -- offset of the value for convolutional matrix
 */

void printUsage()
{
	printf("-conf <conf file>  -- configuration file, the program first read the configuration file options and then use the command line options. That is to say command line options have higher priorities\n");
	printf("-alpha <double>    -- learning rate\n");
	printf("-beta  <double>    -- regulation weight\n");
	printf("-hidden <int>      -- hidden layer size\n");
	printf("-class  <int>      -- class size\n");
	printf("-iter   <int>      -- max iteration\n");
	printf("-fixFirst <bool>   -- true means fix the first layer of the sentence model\n");
	printf("-randInit <bool>   -- true means randomly initialize sentence model word embedding; false means using word2vec to initialize the layer\n");
	printf("-trainF    <train file>\n");
	printf("-validF    <valid file>\n");
	printf("-testF     <test file>\n");
	printf("-embedF    <word embeding file>\n");
	printf("-saveModel <int>  -- 0 do not save model; 1 save every 10000 poems; 2 save at the end\n");
	printf("-modelF    <model file>\n");
	printf("-randSeed  <int>   -- random seed for all the random initialization\n");
	printf("-minImp  <double>   -- minmum improvment for convergence, default 1.0001\n");
	printf("-stableAC <double>  -- everytime when flushing the network, activation in last time hidden layer is set to 'stableAC', default 0.1\n");
	printf("-flushOption <int>  -- 1, flush the network every sentence; 2, flush the network every poem; 3, never flush (only flush at the beginning of training)\n");
	printf("-consynMin <float>  -- minimum value for convolutional matrix\n");
	printf("-consynMax <float>  -- maximum value for convolutional matrix\n");
	printf("-consynOffset <float> -- offset of the value for convolutional matrix, should always be positive [consynMin - consynOffset, consynMax + consynOffset]\n");
	printf("-direct <bool> -- true means the direct error from output layer to condition layer will be used\n");
	printf("-conbptt <bool> -- true means use BPTT training for the recurrent context model (during learning the sentence model)\n");
	printf("-vocabClassF <vocab class path> -- the path of the new vocabulary class; no path mean do not use new vocabulary class\n");
	printf("-historyStableAC <float> -- stableAC in recurrent context model\n");
	printf("-adaGrad <bool> -- true means using AdaGrad training algorithm; false means using SDG\n");
}

void cmdLineConf(int argc, char **argv)
{
	cout << "RNN_Poem_Generation_7-1_CB" << endl;
	if(argc < 3)
	{
		printUsage();
		exit(1);
	}
	// RNNPG rnnpg;
	int pos;
	//const int可以当做常量使用
	const int PATH_LEN = 1024;
	double alpha = 0.05;
	double alphaDiv = 2.0;
	double beta = 0.0000001;
	int hiddenSize = 100;
	int classSize = 100;
	int maxIter = 50;
	bool fixFirst = false;
	bool randInit = false;
	char trainF[PATH_LEN];
	char validF[PATH_LEN];
	char testF[PATH_LEN];
	char embedF[PATH_LEN];
	char modelF[PATH_LEN];
	char vocabClassF[PATH_LEN];
	trainF[0] = 0;
	validF[0] = 0;
	testF[0] = 0;
	embedF[0] = 0;
	modelF[0] = 0;
	vocabClassF[0] = 0;
	int saveModel = 0;
	int randSeed = 1;
	double minImp = 1.0001;
	double stableAC = 0.1;
	int flushOption = 1;		// flush every sentence
	double consynMin = -10;
	double consynMax = 10;
	double consynOffset = 0;
	bool direct = false;
	bool conbptt = false;
	double historyStableAC = 0.01;
	bool adaGrad = false;
	if((pos = getArgPos(argc, argv, "-conf")) != -1)
	{
		Config::load(argv[pos + 1]);
		alpha = Config::getDouble( "alpha" );
		alphaDiv = Config::getDouble( "alphaDiv" );
		beta = Config::getDouble( "beta" );
		hiddenSize = Config::getInt( "hidden" );
		classSize = Config::getInt( "class" );
		maxIter = Config::getInt( "iter" );
		fixFirst = Config::getBool( "fixFirst" );
		randInit = Config::getBool( "randInit" );
		xstrcpy(trainF, sizeof(trainF), Config::getStr("trainF"));
		xstrcpy(validF, sizeof(validF), Config::getStr("validF"));
		xstrcpy(testF, sizeof(testF), Config::getStr("testF"));
		xstrcpy(embedF, sizeof(embedF), Config::getStr("embedF"));
		saveModel = Config::getInt( "saveModel" );
		xstrcpy(modelF, sizeof(modelF), Config::getStr("modelF"));
		randSeed = Config::getInt("randSeed");
		minImp = Config::getDouble("minImp");
		stableAC = Config::getDouble("stableAC");
		flushOption = Config::getInt("flushOption");
		consynMin = Config::getDouble("consynMin");
		consynMax = Config::getDouble("consynMax");
		consynOffset = Config::getDouble("consynOffset");
		direct = Config::getBool("direct");
		conbptt = Config::getBool("conbptt");
		xstrcpy(vocabClassF, sizeof(vocabClassF), Config::getStr("vocabClassF"));
		if(strcmp(vocabClassF, "NO_SUCH_KEY") == 0)
			vocabClassF[0] = 0;

		historyStableAC = Config::getDouble("historyStableAC");
		adaGrad = Config::getBool( "adaGrad" );
	}
	if((pos = getArgPos(argc, argv, "-alpha")) != -1)
		alpha = atof(argv[pos + 1]);

	if((pos = getArgPos(argc, argv, "-alphaDiv")) != -1)
		alphaDiv = atof(argv[pos + 1]);

	if((pos = getArgPos(argc, argv, "-beta")) != -1)
		beta = atof(argv[pos + 1]);

	if((pos = getArgPos(argc, argv, "-hidden")) != -1)
		hiddenSize = atoi(argv[pos + 1]);

	if((pos = getArgPos(argc, argv, "-class")) != -1)
		classSize = atoi(argv[pos + 1]);

	if((pos = getArgPos(argc, argv, "-iter")) != -1)
		maxIter = atoi(argv[pos + 1]);

	if((pos = getArgPos(argc, argv, "-fixFirst")) != -1)
		fixFirst = atob(argv[pos + 1]);

	if((pos = getArgPos(argc, argv, "-randInit")) != -1)
		randInit = atob(argv[pos + 1]);

	if((pos = getArgPos(argc, argv, "-trainF")) != -1)
		xstrcpy(trainF, sizeof(trainF), argv[pos+1]);

	if((pos = getArgPos(argc, argv, "-validF")) != -1)
		xstrcpy(validF, sizeof(validF), argv[pos+1]);

	if((pos = getArgPos(argc, argv, "-testF")) != -1)
		xstrcpy(testF, sizeof(testF), argv[pos+1]);

	if((pos = getArgPos(argc, argv, "-embedF")) != -1)
		xstrcpy(embedF, sizeof(embedF), argv[pos+1]);

	if((pos = getArgPos(argc, argv, "-saveModel")) != -1)
		saveModel = atob(argv[pos+1]);

	if((pos = getArgPos(argc, argv, "-modelF")) != -1)
		xstrcpy(modelF, sizeof(modelF), argv[pos+1]);

	if((pos = getArgPos(argc, argv, "-randSeed")) != -1)
		randSeed = atoi(argv[pos+1]);

	if((pos = getArgPos(argc, argv, "-minImp")) != -1)
			minImp = atof(argv[pos+1]);

	if((pos = getArgPos(argc, argv, "-stableAC")) != -1)
		stableAC = atof(argv[pos+1]);

	if((pos = getArgPos(argc, argv, "-flushOption")) != -1)
		flushOption = atoi(argv[pos+1]);

	if((pos = getArgPos(argc, argv, "-consynMin")) != -1)
		consynMin = atof(argv[pos+1]);

	if((pos = getArgPos(argc, argv, "-consynMax")) != -1)
		consynMax = atof(argv[pos+1]);

	if((pos = getArgPos(argc, argv, "-consynOffset")) != -1)
		consynOffset = atof(argv[pos+1]);

	if((pos = getArgPos(argc, argv, "-direct")) != -1)
		direct = atob(argv[pos + 1]);

	if((pos = getArgPos(argc, argv, "-conbptt")) != -1)
		conbptt = atob(argv[pos + 1]);

	if((pos = getArgPos(argc, argv, "-vocabClassF")) != -1)
		xstrcpy(validF, sizeof(validF), argv[pos+1]);

	if((pos = getArgPos(argc, argv, "-historyStableAC")) != -1)
		historyStableAC = atof(argv[pos+1]);

	if((pos = getArgPos(argc, argv, "-adaGrad")) != -1)
		adaGrad = atob(argv[pos+1]);

	RNNPG rnnpg; //rnnpg部分
	//这里是一些参数初始化的工作
	rnnpg.setAlpha(alpha);
	rnnpg.setAlphaDiv(alphaDiv);
	rnnpg.setBeta(beta);
	rnnpg.setHiddenSize(hiddenSize);
	rnnpg.setClassSize(classSize);
	rnnpg.setMaxIter(maxIter);
	rnnpg.setFixSentenceModelFirstLayer(fixFirst);
	rnnpg.setTrainFile(trainF);
	rnnpg.setValidFile(validF);
	rnnpg.setTestFile(testF);
	rnnpg.setWordEmbeddingFile(embedF);
	rnnpg.setSaveModel(saveModel);
	rnnpg.setModelFile(modelF);
	rnnpg.setRandomSeed(randSeed);
	rnnpg.setRandomlyInitSenModelEmbedding(randInit);
	rnnpg.setMinImprovement(minImp);
	rnnpg.setStableAC(stableAC);
	rnnpg.setFlushOption(flushOption);
	rnnpg.setConsynMin(consynMin);
	rnnpg.setConsynMax(consynMax);
	rnnpg.setConsynOffset(consynOffset);
	rnnpg.setDirectError(direct);
	rnnpg.setConbptt(conbptt);

	rnnpg.setHistoryStableAC(historyStableAC);

	// this is for test, turned off
	rnnpg.setPerSentUpdate(false);

	// this is for test, no improvments yet. Turn off!
	rnnpg.setAdaGrad(false);

	if(vocabClassF[0] != 0)
		rnnpg.setVocabClassFile(vocabClassF);

	//这里开始训练了，核心的部分应该在这里
	if(trainF[0] != 0 && validF[0] != 0 && testF[0] != 0)
		rnnpg.trainNet();

	if(trainF[0] == 0 && validF[0] == 0 && testF[0] != 0)
		rnnpg.testNet();
}

//这个函数没有用到
void testDecoder()
{
	const char *modelPath = "/afs/inf.ed.ac.uk/user/s12/s1270921/Desktop/programming/c++/RNN-Related/RNN_Poem_Generation_Decoder/models/iter066_74809_final";
	RNNPG rnnpg;
	rnnpg.loadNet(modelPath);
	TranslationTable transTable;
	transTable.load("/disk/scratch/RNN_POEM/Data/quatrain/translation-table/trans_tabl_L1.txt");

	Decoder decoder(&rnnpg, NULL);
	vector<string> prevSents;
	int stackSize = 300, topK = 300;
//	prevSents.push_back("床 前 明 月 光");
//	prevSents.push_back("疑 是 地 上 霜");
//	prevSents.push_back("举 头 望 明 月");

//	prevSents.push_back("江 枫 渔 火 对 愁 眠");
//	prevSents.push_back("孤 苏 城 外 寒 山 寺");

//	prevSents.push_back("月 落 乌 啼 霜 满 天");
//	prevSents.push_back("江 枫 渔 火 对 愁 眠");
//	prevSents.push_back("孤 苏 城 外 寒 山 寺");

	cout << "start" << endl;
	clock_t start = clock();
//	prevSents.push_back("春 眠 不 觉 晓");
//	prevSents.push_back("处 处 闻 啼 鸟");
//	prevSents.push_back("夜 来 风 雨 声");
	prevSents.push_back("夜 来 风 雨 R");

	vector<string> topSents;
	decoder.decode(prevSents, stackSize, topK, topSents);
	for(int i = 0; i < (int)topSents.size(); i ++)
		cout << topSents[i] << endl;
	clock_t end = clock();
	cout << "time spend " << (double)(end - start)/CLOCKS_PER_SEC << " S" << endl;
}

//这个函数没有用到
void printUsageDecoder()
{
	printf("-inFile <infile>\n");
	printf("-outFile <outfile>\n");
	printf("-model <modelPath>\n");
	printf("-transTable  <translation table path>\n");
	printf("-stackSize <int>  -- stack size during decoding\n");
	printf("-topK  <int>      -- topK ranked decoded sentences will be returned\n");
	printf("-startId   <int>  -- start id for the decoder, the first column of the returned result is for the sentence id\n");
	printf("-channelOption <int> -- 1 for P(S_i | F_i); 2 for P(F_i | S_i); 3 for P(S_i | F_i) and P(F_i | S_i); 0 for only use the translation table for phrase selection\n");
	printf("-rerank <int> -- 1 for re-rank results with RNNPG values; 0 for not\n");
	printf("-disableRNN <int> -- 1 disable RNN during decoding; 0 for not\n");
	printf("-weightFile <weightFile>\n");
}

//这个函数没有用到
void cmdLineDecoder(int argc, char **argv)
{
	if(argc < 3)
	{
		printUsageDecoder();
		exit(1);
	}

	int pos = -1;
	const int PATH_LENGTH = 1024;
	char infile[PATH_LENGTH];	infile[0] = 0;
	char outfile[PATH_LENGTH];	outfile[0] = 0;
	char modelPath[PATH_LENGTH]; modelPath[0] = 0;
	char transTablePath[PATH_LENGTH]; transTablePath[0] = 0;
	char weightPath[PATH_LENGTH]; weightPath[0] = 0;
	int stackSize = 300;
	int topK = 300;
	int startId = 0;
	int channelOption = 1;
	int rerank = 0;
	int disableRNN = 0;

	if((pos = getArgPos(argc, argv, "-inFile")) != -1)
		xstrcpy(infile, sizeof(infile), argv[pos+1]);

	if((pos = getArgPos(argc, argv, "-outFile")) != -1)
		xstrcpy(outfile, sizeof(outfile), argv[pos+1]);

	if((pos = getArgPos(argc, argv, "-model")) != -1)
		xstrcpy(modelPath, sizeof(modelPath), argv[pos+1]);

	if((pos = getArgPos(argc, argv, "-transTable")) != -1)
		xstrcpy(transTablePath, sizeof(transTablePath), argv[pos+1]);

	if((pos = getArgPos(argc, argv, "-stackSize")) != -1)
		stackSize = atoi(argv[pos+1]);

	if((pos = getArgPos(argc, argv, "-topK")) != -1)
		topK = atoi(argv[pos+1]);

	if((pos = getArgPos(argc, argv, "-startId")) != -1)
		startId = atoi(argv[pos+1]);

	if((pos = getArgPos(argc, argv, "-channelOption")) != -1)
		channelOption = atoi(argv[pos+1]);

	if((pos = getArgPos(argc, argv, "-rerank")) != -1)
		rerank = atoi(argv[pos+1]);

	if((pos = getArgPos(argc, argv, "-disableRNN")) != -1)
		disableRNN = atoi(argv[pos+1]);

	if((pos = getArgPos(argc, argv, "-weightFile")) != -1)
		xstrcpy(weightPath, sizeof(weightPath), argv[pos+1]);

	RNNPG rnnpg;
	rnnpg.loadNet(modelPath);
	TranslationTable *transTable = NULL;
	if(transTablePath[0] != 0)
	{
		transTable = new TranslationTable();
		transTable->load(transTablePath);
	}
	Decoder decoder(&rnnpg, transTable);
	if(weightPath[0] != 0)
		decoder.loadWeights(weightPath);
	decoder.setChannelOption(channelOption);
	decoder.setRerank(rerank);
	decoder.setDisableRNN(disableRNN);
	if(transTable == NULL)
		decoder.decode(infile, outfile, stackSize, topK, startId);
	else
		decoder.decodeTransTable(infile, outfile, stackSize, topK, startId);

	if(transTable != NULL) delete transTable;
}

int main(int argc, char **argv)
{
	// cmdLineDecoder(argc, argv);
	cmdLineConf(argc, argv);

	return 0;
}
