/*
 * ToneRhythm.cpp
 *
 *  Created on: 2014年3月12日
 *      Author: xing
 */


#include "ToneRhythm.h"

const char * const ToneRhythm::tone_strs[] = { "上平", "下平", "上声", "去声", "入声" };
const char * const ToneRhythm::psyFiles[] = {"ShangPing", "XiaPing", "ShangSheng", "QuSheng", "RuSheng"};
const int ToneRhythm::FILE_COUNT = 5;


/**
 * @brief
 * 载入平水韵
 * @param psyTableD 平水韵文件夹的路径
 */
void ToneRhythm::loadAll(string psyTableD)
{
	ancCharDicts.clear();
	int i;
	for(i = 0; i < FILE_COUNT; i ++)
	{
		string path = psyTableD + "/" + psyFiles[i];
		ifstream fin(path.c_str());
		if(!fin.is_open())
		{
			cerr << "open file " << path << " failed!" << endl;
			continue;
		}
		string line;
		map<string, vector<AncChar> > ancCharDict;//对每个文件建立一个ancCharDict
		ancCharDicts.push_back(ancCharDict);//vector<map<string, vector<AncChar>>>
		int lastIndex = ancCharDicts.size() - 1;
		while(getline(fin, line))
		{
			vector<string> fields;
			split(line, "\t", fields);
			//举个例子，原文：去声	一	送	送 梦 凤 洞 众 瓮 贡 弄 冻 痛 栋 恸 仲 中 粽 讽 空 控 哄 赣
			//fields[0]:去声
			//fields[1]:一
			//fields[2]:送
			//fields[3]:送 梦 凤 洞 众 瓮 贡 弄 冻 痛 栋 恸 仲 中 粽 讽 空 控 哄 赣
			int tone = getToneIndex(fields[0]);
			string numstr = fields[1];
			string reprChar = fields[2];
			vector<string> chars;
			split(fields[3], " ", chars);
			//以上面的例子来说
			for(int i = 0; i < chars.size(); i ++)
			{
				string ch = chars[i];
				AncChar ancChar;
				ancChar.ch = ch;//送/梦/...
				ancChar.tone = tone;//3
				ancChar.numstr = numstr;//一
				ancChar.reprChar = reprChar;//送

				map<string, vector<AncChar> >::iterator iter =
						ancCharDicts[lastIndex].find(ch);
				//如果找不到
				if(iter == ancCharDicts[lastIndex].end())
				{
					vector<AncChar> v;
					v.push_back(ancChar);
					ancCharDicts[lastIndex][ch] = v;
				}//如果找到了
				else
					iter->second.push_back(ancChar);
			}
		}
		cout << "load " << path << " done, totally " << ancCharDicts[lastIndex].size() << endl;
	}
}


/**
 * return
 * Ping 1 (0b1)上平/下平
 * Ze 2 (0b10)上声/入声
 * Ping and Ze 3 (0b11)去声
 * Not recorded (0b00)没有在平水韵里出现的词
 * Note when this character is not recorded in ShuiPingYun,
 * then the function return 0b00
 * @param ch
 * @return
 */
int ToneRhythm::getTone(const string &ch)
{
	int ret = 0;
	for(int i = 0; i < ancCharDicts.size(); i ++)
	{
		map<string, vector<AncChar> >::iterator iter = ancCharDicts[i].find(ch);
		if(iter != ancCharDicts[i].end())
			ret = i < 2 ? (ret | 1) : (ret | 2);
	}
	return ret;
}

/**
 * @brief
 * 返回平水韵中对应的ancChar
 * @param ch 查询的字
 * @param ancChar 对应的ancChar
 * @return bool 如果查到了返回true，查不到返回false
 */
bool ToneRhythm::getRhythm(const string &ch, AncChar &ancChar)
{
	for(int i = 0; i < ancCharDicts.size(); i ++)
	{
		map<string, vector<AncChar> >::iterator iter = ancCharDicts[i].find(ch);
		if(iter != ancCharDicts[i].end())
		{
			ancChar = iter->second[0];
			return true;
		}
	}

	return false;
}



