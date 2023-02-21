#include <iostream>
#include <cstdio>
#include <cstring>
using namespace std;

//保留字
string key[6] = {"main", "int", "if", "else", "while", "do"};

//保留字的种别码
int keyNum[6] = {1, 2, 3, 4, 5, 6};

//运算符和界符
string symbol[17] = {"<", ">", "!=", ">=", "<=", "==", ",", ";", "(", ")", "{", "}", "+", "-", "*", "/", "="};

//运算符和界符的种别码
int symbolNum[17] = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};

//存放文件取出的字符
string letter[1000];

//将字符转换为单词
string words[1000];

int length; //保存程序中字符的数目
int num;

int isSymbol(string s) //判断运算符和界符
{
	int i;
	for (i = 0; i < 17; i++)
	{
		if (s == symbol[i])
			return symbolNum[i];
	}
	return 0;
}

bool isNum(string s) //判断是否为数字
{
	if (s >= "0" && s <= "9")
		return true;
	return false;
}

bool isLetter(string s) //判断是否为字母
{
	if (s >= "a" && s <= "z")
		return true;
	return false;
}

int isKeyWord(string s) //判断是否为关键字,是返回种别码
{
	int i;
	for (i = 0; i < 6; i++)
	{
		if (s == key[i])
			return keyNum[i];
	}
	return 0;
}


int wordtype(string str) //返回首字符的类型
{
	if (str >= "a" && str <= "z") //字母
		return 1;

	if (str >= "0" && str <= "9") //数字
		return 2;

	if (str == ">" || str == "=" || str == "<" || str == "!" 
		|| str == "," || str == ";" || str == "(" || str == ")" 
			|| str == "{" || str == "}" || str == "+" || str == "-" 
				|| str == "*" || str == "/") //判断运算符和界符
		return 3;
}

string judge(string s, int n) //判断单词是保留字还是标识符
{
	int j = n + 1;
	int flag = 1;

	while (flag)
	{
		if (isNum(letter[j]) || isLetter(letter[j])) //字符是数字或者是字母
		{
			s = (s + letter[j]).c_str();
			if (isKeyWord(s))
			{
				j++;
				num = j;
				return s;
			}
			j++;
		}
		else
		{
			flag = 0;
		}
	}

	num = j;
	return s;
}

string symbolStr(string s, int n)
{
	int j = n + 1;
	string str = letter[j];
	if (str == ">" || str == "=" || str == "<" || str == "!")
	{
		s = (s + letter[j]).c_str();
		j++;
	}
	num = j;
	return s;
}

string Number(string s, int n)
{
	int j = n + 1;
	int flag = 1;

	while (flag)
	{
		if (isNum(letter[j]))
		{
			s = (s + letter[j]).c_str();
			j++;
		}
		else
		{
			flag = 0;
		}
	}

	num = j;
	return s;
}

void print(string s, int n)
{
	cout << "(" << s << "," << n << ")" << endl;
}

void reading() //取单词
{
	int k;

	for (num = 0; num < length;)
	{
		string str1, str;
		str = letter[num];
		k = wordtype(str);
		switch (k)
		{
		case 1: //标识符或者保留字
		{
			str1 = judge(str, num);
			if (isKeyWord(str1))
				print(str1, isKeyWord(str1));
			else
				print(str1, 0);
			break;
		}

		case 2: //数字
		{
			str1 = Number(str, num);
			print(str1, 24);
			break;
		}

		case 3: //符号
		{
			str1 = symbolStr(str, num);
			print(str1, isSymbol(str1));
			break;
		}
		}
	}
}

int main()
{
	char w;
	int i, j;

	freopen("input.txt", "r", stdin);

	length = 0;
	while (cin >> w) //把文件内容转存到letter字符数组中
	{
		if (w != ' ')
		{
			letter[length] = w;
			length++;
		} 
	}

	reading();

	fclose(stdin);	//关闭文件
	return 0;
}
