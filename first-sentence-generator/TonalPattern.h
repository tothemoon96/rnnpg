/*
 * TonalPattern.h
 *
 *  Created on: 13 Mar 2014
 *      Author: s1270921
 */

#ifndef TONALPATTERN_H_
#define TONALPATTERN_H_

#include <iostream>
#include <string>
#include <stdio.h>
using namespace std;
#include "xutil.h"

const char * const _STD_5QUATRAIN_TPAT_[] = {
	"AZPPZ\nPPZZP\nAPPZZ\nAZZPP\n",
	"AZZPP\nPPZZP\nAPPZZ\nAZZPP\n",
	"APPZZ\nAZZPP\nAZPPZ\nPPZZP\n",
	"PPZZP\nAZZPP\nAZPPZ\nPPZZP\n"
};

const char * const _STD_7QUATRAIN_TPAT_[] = {
	"APAZZPP\nAZPPZZP\nAZAPPZZ\nAPAZZPP\n",
	"APAZPPZ\nAZPPZZP\nAZAPPZZ\nAPAZZPP\n",
	"AZPPZZP\nAPAZZPP\nAPAZPPZ\nAZPPZZP\n",
	"AZAPPZZ\nAAZZZPP\nAPAZPPZ\nAZPPZZP\n"
};

/**
 * @brief
 * 存储一句诗的韵律
 */
class SenTP
{
public:
	int tonalPattern;
	int validPos;
	int senLen;

	/**
	 * @brief
	 * 解析整个句子的韵律，转换成二进制的位表示
	 * 举个例子:
	 * pat:"APAZZPP"
	 * 和字符串的顺序反过来的
	 * 分离P与Z,Z为1,P为0->tonalPattern(bit):0011000
	 * 分离PZ与A,PZ为1,A为0->validPos(bit):  1111010
	 * @param pat 存储韵律的字符串
	 */
	void parse(const string &pat)
	{
		tonalPattern = 0;
		validPos = 0;

		senLen = pat.length();
		int base = 1;
		int preValidPos = 0;
		for(int i = 0; i < (int)pat.length(); i ++)
		{
			char ch = pat[i];
			if(ch == 'Z') // p 0, z 1
			{
				tonalPattern = tonalPattern | base;
				validPos = preValidPos | base;
			}
			else if(ch == 'P')
				validPos = preValidPos | base;
			base = base << 1;
			preValidPos = validPos;
		}
	}

	/**
	 * @brief
	 * 检测pat代表的二进制韵律是否是有效的韵律
	 * @param pat
	 * @param vPos
	 * @return bool
	 */
	bool isValidPattern(int pat, int vPos)
	{
		return (tonalPattern & validPos & vPos) == (pat & validPos & vPos);
	}

	/**
	 * @brief
	 * 将位表示的韵律还原成字符串表示的韵律
	 * 和字符串的顺序反过来的
	 * 分离P与Z,Z为1,P为0->tonalPattern(bit):0011000
	 * 分离PZ与A,PZ为1,A为0->validPos(bit):  1111010
	 * return:"APAZZPP"
	 * @param tonalPattern 分离P与Z
	 * @param validPos 分离PZ与A
	 * @param senLen 句子的长度
	 * @return string　返回的字符串表示的韵律
	 */
	static string inttp2strtp(int tonalPattern, int validPos, int senLen)
	{
		string tonalStr = "";
		int base = 1;
		for(int i = 0; i < senLen; i ++)
		{
			if((validPos & base) != 0)
			{
				if((tonalPattern & base) != 0)
					tonalStr += "Z";
				else
					tonalStr += "P";
			}
			else
				tonalStr += "A";
			base = base << 1;
		}
		return tonalStr;
	}

	/**
	 * @brief
	 * 将该对象表示的韵律转换成字符串
	 * @return string
	 */
	string toString()
	{
		return inttp2strtp(tonalPattern, validPos, senLen);
	}

	void printPZ()
	{
		int base = 1, i;
		printf("tonalPattern - ");
		for(i = 0; i < senLen; i ++)
		{
			if((tonalPattern & base) != 0)
				printf("Z");
			else
				printf("P");
			base = base << 1;
		}
		putchar('\n');

		printf("validPos - ");
		base = 1;
		for(i = 0; i < senLen; i ++)
		{
			if((validPos & base) != 0)
				printf("Y");
			else
				printf("N");
			base = base << 1;
		}
		putchar('\n');
	}
};

/**
 * @brief
 * 存储一首诗的韵律
 */
class PoemTP
{
public:
	vector<SenTP> poemtp;

	/**
	 * @brief
	 * 添加每句话的韵律
	 * @param pat 描述一句诗韵律的对象
	 */
	void addPat(string pat)
	{
		SenTP sentp;
		sentp.parse(pat);
		poemtp.push_back(sentp);
	}
	/**
	 * @brief
	 * 获得某句诗的韵律
	 * @param i 编号0-3
	 * @return SenTP 韵律对象
	 */
	SenTP& getSenTP(int i)
	{
		return poemtp[i];
	}
};

/**
 * @brief
 * 保存几种通用的格律
 */
class TonalPattern
{
public:
	TonalPattern() {	init();	}
	/**
	 * @brief
	 * 构造对象的时候调用，就是用_STD_5QUATRAIN_TPAT_和_STD_7QUATRAIN_TPAT_构造几种PoemTP
	 */
	void init()
	{
		int i, j;
		// 处理五言诗
		for(i = 0; i < 4; i ++)
		{
			string tpat = _STD_5QUATRAIN_TPAT_[i];
			PoemTP poemtp;
			qua5tps.push_back(poemtp);
			int last = qua5tps.size() - 1;
			vector<string> fields;
			split(tpat, "\n", fields);
			for(j = 0; j < (int)fields.size(); j ++)
				qua5tps[last].addPat(fields[j]);
		}
		//　处理七言诗
		for(i = 0; i < 4; i ++)
		{
			string tpat = _STD_7QUATRAIN_TPAT_[i];
			PoemTP poemtp;
			qua7tps.push_back(poemtp);
			int last = qua7tps.size() - 1;
			vector<string> fields;
			split(tpat, "\n", fields);
			for(j = 0; j < (int)fields.size(); j ++)
				qua7tps[last].addPat(fields[j]);
		}
	}

	/**
	 * @brief
	 * 获得几种韵律格式下第一句诗的韵律，存储在sentps中
	 * @param wordsPerSen 一句诗的长度
	 * @param sentps 存储返回韵律对象的容器
	 */
	void getFirstSenTPs(int wordsPerSen, vector<SenTP> &sentps)
	{
		int i;
		if(wordsPerSen == 5)
		{
			for(i = 0; i < (int)qua5tps.size(); i ++)
				sentps.push_back(qua5tps[i].getSenTP(0));
		}
		else
		{
			for(i = 0; i < (int)qua7tps.size(); i ++)
				sentps.push_back(qua7tps[i].getSenTP(0));
		}
	}

	/**
	 * @brief
	 * 获得第tpIndex韵律格式下第senIndex句诗的韵律
	 * @param wordsPerSen 一句诗的长度
	 * @param tpIndex 编号0-3
	 * @param senIndex 编号0-3
	 * @return SenTP 存储韵律的对象
	 */
	SenTP getSenTP(int wordsPerSen, int tpIndex, int senIndex)
	{
		vector<PoemTP> &quatps = wordsPerSen == 5 ? qua5tps : qua7tps;
		return quatps[tpIndex].getSenTP(senIndex);
	}
private:
	//这两个向量的数据结构
	//TonalPattern(vector<PoemTP>*2)->vector<PoemTP>(PoemTP*4)->PoemTP(SenTP*4)
	vector<PoemTP> qua5tps;//保存了_STD_5QUATRAIN_TPAT_转换成PoemTP的韵律
	vector<PoemTP> qua7tps;//保存了_STD_7QUATRAIN_TPAT_转换成PoemTP的韵律
};

#endif /* TONALPATTERN_H_ */
