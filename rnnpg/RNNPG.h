/*
 * RNNPG.h
 *
 *  Created on: 27 Dec 2013
 *      Author: s1270921
 */

#ifndef __RNNPG_H__
#define __RNNPG_H__

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <utility>
#include <algorithm>
using namespace std;

#include "Vocab.h"
#include "WordEmbedding.h"

// the fast exp() implementation is from Tomas Mikolov's toolkit, rnnlm-0.2b
///// fast exp() implementation
static union{
    double d;
    struct{
        int j,i;
        } n;
} d2i;
#define EXP_A (1048576/M_LN2)
#define EXP_C 60801
//实现了快速exp()的功能
#define FAST_EXP(y) (d2i.n.i = EXP_A*(y)+(1072693248-EXP_C),d2i.d)

bool pair_cmp(const pair<string,double> &p1, const pair<string,double> &p2);
//{
//	return !(p1.second < p2.second);
//}

//struct PairCmp {
//	bool operator() (const pair<string,double> &p1, const pair<string,double> &p2)
//	{
//	  return !(p1.second < p2.second);
//	}
//}pairCmp;

struct neuron {
	// f(net) or net
    double ac;		//actual value stored in neuron
    double er;		//error value in neuron, used by learning algorithm

    void set(double a = 0, double e = 0)
    {
    	ac = a;
    	er = e;
    }
    void fun_ac()
    {
    	if(ac > 50) ac = 50;
    	if(ac < -50) ac = -50;
    	// sigmoid function
    	ac = 1/(1+FAST_EXP(-ac));
    }
};

//神经元之间连接的权值
struct synapse {
    double weight;	//weight of synapse
};

//卷积层的数目
const int MAX_CON_N = 4;
//参见论文里卷积神经网络C每一层卷积核大小
const int conSeq[MAX_CON_N] = { 2, 2, 3, 3 };  // convolution sequence
const int MAX_PATH_LENGTH = 256;

class RNNPG
{
public:
	RNNPG();
	~RNNPG();
	void initNet();
	void trainNet();
	void testNet();
	void loadNet(const char *infile);
	void saveNet(const char *outfile);
	void loadVocab(const char *trainFile);
	void loadVocabFromTrain() { loadVocab(trainFile); }
	void getContextHiddenNeu(vector<string> &prevSents, neuron **contextHiddenNeu);
	void computeNetContext(const char *lword, const char *cword, neuron *curHiddenNeu, neuron *contextHiddenNeu,
			neuron *newHiddenNeu, vector<pair<string,double> > &nextWordProbs);
	double computeNetContext(const char *lword, int startPos, const vector<string> &words, neuron *curHiddenNeu, neuron **contextHiddenNeus,
			neuron *newHiddenNeu);

	// set parameters
	void setHiddenSize(int size) { hiddenSize = size; }
	void setMaxIter(int iter) { maxIter = iter; }
	void setRandomSeed(int seed) { randomSeed = seed; srand(randomSeed); }
	void setWordEmbeddingFile(const char *wordEmbedFile) { snprintf(wordEmbeddingFile,sizeof(wordEmbeddingFile), "%s",wordEmbedFile); }
	void setTrainFile(const char *trainF) { snprintf(trainFile,sizeof(trainFile), "%s",trainF); }
	void setValidFile(const char *validF) { snprintf(validFile,sizeof(validFile), "%s",validF); }
	void setTestFile(const char *testF) { snprintf(testFile,sizeof(testFile), "%s",testF); }
	void setModelFile(const char *modelF) { snprintf(modelFile,sizeof(modelFile), "%s",modelF); }
	void setVocabClassFile(const char *vclassF) { snprintf(vocabClassF,sizeof(vocabClassF), "%s",vclassF); }
	void setSaveModel(int save) { saveModel = save; }
	void setClassSize(int size) { classSize = size; }
	void setAlpha(double al) { alpha = al; }
	void setAlphaDiv(double aldiv) { alphaDiv = aldiv; }
	void setBeta(double be) { beta = be; }
	void setFixSentenceModelFirstLayer(bool fix) { fixSentenceModelFirstLayer = fix; }
	void setRandomlyInitSenModelEmbedding(bool randomlyInit) { randomlyInitSenModelEmbedding = randomlyInit; }
	void setMinImprovement(double minImp) { minImprovement = minImp; }
	void setStableAC(double ac) { stableAC = ac; }
	void setFlushOption(int option) { flushOption = option; }
	void setConsynMin(double cmin) { consynMin = cmin; }
	void setConsynMax(double cmax) { consynMax = cmax; }
	void setConsynOffset(double offset) { consynOffset = offset; }
	void setDirectError(bool direct) { directError = direct; }
	void setConbptt(bool cbptt) { conbptt = cbptt; }
	void setPerSentUpdate(bool sentUpdate) { perSentUpdate = sentUpdate; }
	void setHistoryStableAC(double ac) { historyStableAC = ac; }
	void setAdaGrad(bool adag) { adaGrad = adag; }

	// get parameters
	int getHiddenSize()	{	return hiddenSize;	}
	void getHiddenNeuron(neuron * hiddenNeurons)
	{
		for(int i = 0; i < hiddenSize; i ++)
		{
			hiddenNeurons[i].ac = hiddenNeu[i].ac;
			hiddenNeurons[i].er = hiddenNeu[i].er;
		}
	}
	int getFlushOption() { return flushOption; }

	// show parameters
	void showParameters();
private:
	//隐含层单元的数目
	int hiddenSize;
	Vocab vocab;
	char trainFile[MAX_PATH_LENGTH];
	char validFile[MAX_PATH_LENGTH];
	char testFile[MAX_PATH_LENGTH];
	char wordEmbeddingFile[MAX_PATH_LENGTH];
	char modelFile[MAX_PATH_LENGTH];
	char vocabClassF[MAX_PATH_LENGTH];
	int randomSeed;
	WordEmbedding wdEmbed;
	synapse *conSyn[MAX_CON_N];			 	 // convolution matrix，这里是一个指针数组
	synapse *conSynOffset[MAX_CON_N];		 // when update convolution matrix, we need to compute the offset first and then add the L2 norm term
	enum SEN_LENGTH {SEN5_LENGTH = 5, SEN7_LENGTH = 7};//对应的是诗歌句子的长度
	enum SEN_TREE_HIGHT{SEN5_HIGHT = 4, SEN7_HIGHT = 5};//CSM：五言诗只有4层，七言诗有5层
	neuron *sen7Neu[SEN7_HIGHT];//CSM网络中对应于7言诗的神经元
	neuron *sen5Neu[SEN5_HIGHT];//CSM网络中对应于5言诗的神经元
	synapse *compressSyn;		// used to compress history representation and the current representation. Or generate new history_i using history_i-1 and sen_repr_i，对应于RCM中的M矩阵
	neuron *hisNeu;		// history representation，对应RCM中的h_i
	neuron *cmbNeu;		// previous history representation and the current representation，对应RCM中的\begin{bmatrix}h_{i-1}\\v_i\end{bmatrix}
	synapse *map7Syn[8];	// this matrix is used to map representation to each postion in the generated sentence (for 7 character sentences)，对应于RCM中7言诗的U_j
	synapse *map5Syn[6];	// this matrix is used to map representation to each postion in the generated sentence (for 5 character sentences)，对应于RCM中5言诗的U_j
	neuron *conditionNeu;	// conditional representation for each sentence，对应于RCM中的u_i^j

	synapse *senweSyn;		// the word embedding matrix in sentence model. This can be modified during training. We can inilizate it with word2vec word embedding, or just randomly

	neuron *inNeu;//对应于RGM中的输入层，其中0到V-1存储的是词的one-hot，V到V+hiddenSize-1存储的是u_i^j，V+hiddenSize到V+hiddenSize*2-1存储的是r_{j-1}
	neuron *hiddenNeu;//对应于RGM的隐含层r_j
	neuron *outNeu;//前V个神经元表示的是P(word|word_class,context)，后面classSize个神经元表示的是P(word_class,context)
	//hiddenInSyn的存储结构
	// |-----V(word embedding矩阵)------|---hiddenSize(H)---|---hiddenSize(R)---|
	//h|								|					|				    |
	//i|								|					|				    |
	//i|								|					|				    |
	//d|								|					|				    |
	//d|								|					|				    |
	//e|								|					|				    |
	//n|								|					|				    |
	//S|								|					|				    |
	//i|								|					|				    |
	//z|								|					|				    |
	//e|								|					|				    |
	synapse *hiddenInSyn;
	//outHiddenSyn的存储结构，大致对应于RGM的Y矩阵
	//|----------hiddenSize--------|
	//|							   |
	//|							   |
	//|							   |用于计算P(word|word_class,context)
	//V							   |
	//|							   |
	//|							   |
	//|							   |
	//-----------------------------|
	//|							   |
	//|							   |
	//|							   |
	//c							   |
	//l							   |
	//a							   |
	//s							   |用于计算P(word_class,context)
	//s							   |
	//S							   |
	//i							   |
	//z							   |
	//e							   |
	//|							   |
	//|							   |
	//|							   |
	synapse *outHiddenSyn;
	int classSize;
	Word *voc_arr;
	int *classStart;//一个int数组，数组中的每个元素对应于V中该类词的开始的编号
	int *classEnd;//一个int数组，数组中的每个元素对应于V中该类词结束的编号+1

	// for back propagation
	int *bpttHistory;
	neuron* bpttHiddenNeu;
	neuron* bpttInHiddenNeu;
	synapse *bpttHiddenInSyn;
	neuron* bpttConditionNeu;

	// for directErr propagate to input layer
	//outConditionDSyn的存储结构，和outHiddenSyn相似，只是不考虑RGM的作用了
	//|----------hiddenSize--------|
	//|							   |
	//|							   |
	//|							   |
	//V							   |
	//|							   |
	//|							   |
	//|							   |
	//-----------------------------|
	//|							   |
	//|							   |
	//|							   |
	//c							   |
	//l							   |
	//a							   |
	//s							   |
	//s							   |
	//S							   |
	//i							   |
	//z							   |
	//e							   |
	//|							   |
	//|							   |
	//|							   |
	synapse *outConditionDSyn;
	neuron *bufOutConditionNeu;		// errors from every word in the sentence to condition neuron directly,u_i^j的缓存

	// for BPTT of recurrent context model
	bool isLastSentOfPoem;
	bool conbptt;
	int contextBPTTSentNum;
	neuron* conBPTTHis;
	neuron* conBPTTCmbHis;
	neuron* conBPTTCmbSent;
	synapse *bpttHisCmbSyn;

	int maxIter;
	double alpha;	// this is the learning rate
	double alphaDiv;	// this is the learning rate decay
	double beta;	// this is the weight for L2 normalization term in objective function,L2正则化因子

	int mode;
	enum ModeType{TRAIN_MODE, TEST_MODE, UNDEFINED_MODE};

	// for controls
	bool firstTimeInit;
	int wordCounter;
	int totalPoemCount;
	bool fixSentenceModelFirstLayer;
	bool randomlyInitSenModelEmbedding;
	double logp;
	double minImprovement;
	int alphaDivide;
	int saveModel;
	double stableAC;
	double historyStableAC;
	int flushOption;
	enum FLUSH_OPTION{EVERY_SENTENCE = 1, EVERY_POEM = 2, NEVER = 3};
	double consynMin;
	double consynMax;
	double consynOffset;
	bool directError;//如果为1，这个好像是控制直接使用RCM去生成，如果为0，考虑RCM和RGM
	bool perSentUpdate;

	// backups
	synapse *conSyn_backup[MAX_CON_N];			 	 // convolution matrix	// backup
	synapse *conSynOffset_backup[MAX_CON_N];		 // when update convolution matrix, we need to compute the offset first and then add the L2 norm term	// backup
	neuron *sen7Neu_backup[SEN7_HIGHT];	// backup
	neuron *sen5Neu_backup[SEN5_HIGHT];	// backup
	synapse *compressSyn_backup;		// used to compress history representation and the current representation. Or generate new history_i using history_i-1 and sen_repr_i	// backup
	neuron *hisNeu_backup;		// history representation	// backup
	neuron *cmbNeu_backup;		// previous history representation and the current representation	// backup
	synapse *map7Syn_backup[8];	// this matrix is used to map representation to each postion in the generated sentence (for 7 character sentences)	// backup
	synapse *map5Syn_backup[6];	// this matrix is used to map representation to each postion in the generated sentence (for 5 character sentences)	// backup
	neuron *conditionNeu_backup;	// conditional representation for each sentence	// backup
	synapse *senweSyn_backup;		// the word embedding matrix in sentence model. This can be modified during training. We can inilizate it with word2vec word embedding, or just randomly	// backup
	neuron *inNeu_backup;	// backup
	neuron *hiddenNeu_backup;	// backup
	neuron *outNeu_backup;	// backup
	synapse *hiddenInSyn_backup;	// backup
	synapse *outHiddenSyn_backup;	// backup
	synapse *outConditionDSyn_backup;		// backup
	// neuron *bufOutConditionNeu_backup;		// backup

	/**
	 * This is for adagrad
	 */
	struct SumGradSquare
	{
		// double sumGradSqInitVal;
		double *conSyn_[MAX_CON_N];
		double *compressSyn_;
		double *map7Syn_[8];
		double *map5Syn_[6];
		double *senweSyn_;
		double *hiddenInSyn_;
		double *outHiddenSyn_;

//		RNNPG *rnnpg;
//
//		SumGradSquare(RNNPG *_rnnpg) : rnnpg(_rnnpg)
//		{
//			int i;
//			for(i = 0; i < MAX_CON_N; i ++)
//				conSyn_[i] = NULL;
//			compressSyn_ = NULL;
//			for(i = 0; i < 8; i ++)
//				map7Syn_[i] = NULL;
//			for(i = 0; i < 6; i ++)
//				map5Syn_[i] = NULL;
//			senweSyn_ = NULL;
//			hiddenInSyn_ = NULL;
//			outHiddenSyn_= NULL;
//		}

		void init(RNNPG *rnnpg)
		{
			int hiddenSize = rnnpg->hiddenSize;
			double sumGradSqInitVal = 0;
			int i, N, M;
			for(i = 0; i < MAX_CON_N; i ++)
			{
				N = hiddenSize * conSeq[i];
				conSyn_[i] = new double[N];
				fill(conSyn_[i], N, sumGradSqInitVal);
			}

			N = hiddenSize * 2 * hiddenSize;
			compressSyn_ = new double[N];
			fill(compressSyn_, N, sumGradSqInitVal);

			N = hiddenSize * hiddenSize;
			M = 8;
			for(i = 0; i < M; i ++)
			{
				map7Syn_[i] = new double[N];
				fill(map7Syn_[i], N, sumGradSqInitVal);
			}
			M = 6;
			for(i = 0; i < M; i ++)
			{
				map5Syn_[i] = new double[N];
				fill(map5Syn_[i], N, sumGradSqInitVal);
			}

			int V = rnnpg->vocab.getVocabSize();
			N = V * hiddenSize;
			senweSyn_ = new double[N];
			fill(senweSyn_, N, sumGradSqInitVal);

			N = hiddenSize * (V + hiddenSize + hiddenSize);
			hiddenInSyn_ = new double[N];
			fill(hiddenInSyn_, N, sumGradSqInitVal);

			int classSize = rnnpg->classSize;
			N = (V + classSize) * hiddenSize;
			outHiddenSyn_ = new double[N];
			fill(outHiddenSyn_, N, sumGradSqInitVal);
		}

		void reset(RNNPG *rnnpg)
		{
			int hiddenSize = rnnpg->hiddenSize;
			double sumGradSqInitVal = 0;
			int i, N, M;
			for(i = 0; i < MAX_CON_N; i ++)
			{
				N = hiddenSize * conSeq[i];
				fill(conSyn_[i], N, sumGradSqInitVal);
			}

			N = hiddenSize * 2 * hiddenSize;
			fill(compressSyn_, N, sumGradSqInitVal);

			N = hiddenSize * hiddenSize;
			M = 8;
			for(i = 0; i < M; i ++)
				fill(map7Syn_[i], N, sumGradSqInitVal);
			M = 6;
			for(i = 0; i < M; i ++)
				fill(map5Syn_[i], N, sumGradSqInitVal);

			int V = rnnpg->vocab.getVocabSize();
			N = V * hiddenSize;
			fill(senweSyn_, N, sumGradSqInitVal);

			N = hiddenSize * (V + hiddenSize + hiddenSize);
			fill(hiddenInSyn_, N, sumGradSqInitVal);

			int classSize = rnnpg->classSize;
			N = (V + classSize) * hiddenSize;
			fill(outHiddenSyn_, N, sumGradSqInitVal);
		}

		void fill(double *arr, int size, double val)
		{
			for(int i = 0; i < size; i ++)
				arr[i] = val;
		}
		void releaseMemory()
		{
			int i, M;
			for(i = 0; i < MAX_CON_N; i ++)
				delete []conSyn_[i];
			delete []compressSyn_;

			M = 8;
			for(i = 0; i < M; i ++)
				delete []map7Syn_[i];
			M = 6;
			for(i = 0; i < M; i ++)
				delete []map5Syn_[i];

			delete []senweSyn_;

			delete []hiddenInSyn_;

			delete []outHiddenSyn_;
		}
	}sumGradSquare;
	bool adaGrad;
	double adaGradEps;

	// private functions
	neuron* sen2vec(const vector<string> &senWords, neuron **senNeu, int SEN_HIGHT);
	void trainPoem(const vector<string> &sentences);
	void testPoem(const vector<string> &sentences);
	void computeNet(int lastWord, int curWord, int wordPos, synapse** mapSyn);
	void learnNet(int lastWord, int curWord, int wordPos, int senLen);
	void learnNetAdaGrad(int lastWord, int curWord, int wordPos, int senLen);
	void learnSent(int senLen);
	void learnSentBPTT(int senLen);
	void learnSentAdaGrad(int senLen);
	void saveWeights();
	void restoreWeights();
	void saveSynapse(FILE *fout);
	void loadSynapse(FILE *fin);
	void saveNeuron(FILE *fout);
	void loadNeuron(FILE *fin);
	void saveBasicSetting(FILE *fout);
	void loadBasicSetting(FILE *fin);
	// void save
	void initBackup();
    void matrixXvector(struct neuron *dest, struct neuron *srcvec, struct synapse *srcmatrix, int matrix_width, int from, int to, int from2, int to2, int type);
    void initSent(int senLen);
    void flushNet();
    void testNetFile(const char *testF);
    void assignClassLabel();
    void copyHiddenLayerToInput();
	/**
	 * @brief
	 * 清空从neus指针开始，往后size个神经元的数据
	 * @param neus 开始清空的神经元指针
	 * @param size 清空多少个
	 * @param flag 清空模式，flag=1代表的是只清空正向传播的值，flag=2代表清空反向传播的值，flag=3代表正向反向都清空
	 */
	void clearNeurons(neuron* neus, int size, int flag)
	{
		for(int i = 0; i < size; i ++)
		{
			//注意这里做的是与运算，flag=1代表的是只清空正向传播的值，flag=2代表清空反向传播的值，flag=3代表正向反向都清空
			if(flag & 1) neus[i].ac = 0;
			if(flag & 2) neus[i].er = 0;
		}
	}
    void copyNeurons(neuron* dstNeu, neuron* srcNeu, int size, int flag)
    {
    	for(int i = 0; i < size; i ++)
    	{
    		if(flag & 1) dstNeu[i].ac = srcNeu[i].ac;
    		if(flag & 2) dstNeu[i].er = srcNeu[i].er;
    	}
    }
	/**
	 * @brief
	 * 对一个神经元的数组逐个进行激活
	 * @param neus 指向神经元数组的指针
	 * @param size 数组的长度
	 */
	void funACNeurons(neuron* neus, int size)
    {
    	for(int i = 0; i < size; i ++)
    		neus[i].fun_ac();
    }
    void printNeurons(neuron* neus, int size)
    {
    	for(int i = 0; i < size; i ++)
    		cout << "(" << neus[i].ac << "," << neus[i].er << ") ";
    	cout << endl;
    }
	inline bool isSep(char ch)	{	return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r';		}
	/**
	 * @brief
	 * 产生min和max之间的随机数
	 * @param min 最大值
	 * @param max 最小值
	 * @return double 返回的浮点数
	 */
	inline double random(double min, double max)
	{
	    return rand()/(double)RAND_MAX*(max-min)+min;
	}
};

#endif


