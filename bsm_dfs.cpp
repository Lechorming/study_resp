#include <iostream>
#include <cstring>

using namespace std;

struct state
{
	char n[9]; //用字符数组表示棋盘状态
	int d = -1; //状态深度
	int zx, zy; //空格的位置
	state *father = NULL; //父节点指针
};

int jc[10]; //阶乘
state past[362885]; //记录到达过的状态,用于回溯从初始状态到目标状态的路径,close表
int aim; //目标状态 
int mi = 362880;
char ch[30][10];
int cnt = 0;
int pastcnt=0;

void jiecheng()
{
	jc[0] = 0;
	jc[1] = 1;
	for (int i = 2; i < 10; i++)
		jc[i] = jc[i - 1] * i;
}

int kangtuo(char n[]) //康托展开
{
	int ans = 0;
	for (int i = 0; i < 9; i++)
	{
		int cnt = 0;
		for (int j = i + 1; j < 9; j++)
			if (n[i] > n[j])
				cnt++;
		ans += cnt * jc[9 - i - 1];
	}
	return ans;
}

void swap(char &a, char &b)
{
	char t;
	t = a;
	a = b;
	b = t;
}

void dfsrecord(state *p) //记录最短路径
{
	if (p->father)
		dfsrecord(p->father);
	strcpy(ch[cnt++], p->n);
	return;
}

void prtans()
{
	cout << "从初始状态到目标状态需要" << mi << "步\n";
	for (int i = 0; i <= mi; i++)
	{
		for (int x = 0; x < 3; x++)
		{
			for (int y = 0; y < 3; y++)
				cout << ch[i][x * 3 + y] << ' ';
			cout << '\n';
		}
		cout << '\n';
	}
}

int dfs(state s)
{
	pastcnt++; //搜索结点数计数器
	int kt = kangtuo(s.n);
	if ((past[kt].d != -1 && past[kt].d <= s.d))
		return 0;  //当前状态和历史状态深度比较，更深或相等则返回
	past[kt] = s;  //更新历史状态
	if (kt == aim) //到达目标状态
	{
		cnt = 0;
		mi = past[kt].d;
		dfsrecord(&(past[kt]));
		return 1;
	}
	if(s.d+1>30) return 0;
	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 2; j++)
		{
			int x = i ? (j ? s.zx + 1 : s.zx - 1) : s.zx;
			int y = i ? s.zy : (j ? s.zy + 1 : s.zy - 1); // x,y表示扩展状态的空格位置,分别可以向上下左右四个方向扩展
			if (x >= 0 && x <= 2 && y >= 0 && y <= 2 &&
				((s.father ? (x != s.father->zx) : 1) ||
				 (s.father ? (y != s.father->zy) : 1)))
			{
				state temp; //新建一个状态表示扩展状态
				strcpy(temp.n, s.n); //复制父状态到扩展状态
				swap(temp.n[s.zx * 3 + s.zy], temp.n[x * 3 + y]); //交换空格位置
				temp.d = s.d + 1; //扩展状态深度+1
				temp.zx = x;
				temp.zy = y; //更新扩展状态的空格位置
				temp.father = &(past[kt]); //扩展状态的父节点
				dfs(temp); //扩展状态压入open表
			}
		}
	return 0;
}

int main()
{
	state start;
	cout << "输入初始状态矩阵(用0代替棋盘上的空格):\n";
	for (int i = 0; i < 9; i++)
	{
		cin >> start.n[i];
		if (start.n[i] == '0')
		{
			start.zx = i / 3;
			start.zy = i % 3;
		}
	}
	start.d = 0;
	start.father = NULL;
	jiecheng();
	char ac[]="123804765";
	aim = kangtuo(ac);
	dfs(start);
	if (mi == 362880)
		cout << "该初始状态无解！\n";
	else
	{
		prtans();
	}
	cout <<"已扩展状态："<< pastcnt << '\n';
}
