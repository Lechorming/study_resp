#include <iostream>
#include <cstring>
#include <queue>

using namespace std;

struct state
{
	char n[9];			  //用字符数组表示棋盘状态
	int d = 0;			  //状态深度
	int zx, zy;			  //空格的位置
	state *father = NULL; //父节点指针
};

int jc[10];				  //阶乘
queue<state> q;			  //等待队列,open表
bool f[362885];			  //康托展开判重
state past[362885];		  //记录到达过的状态,用于回溯从初始状态到目标状态的路径,close表
int cp = 0;				  //状态计数器
char aim[] = "123804765"; //目标状态
int pastcnt = 0;

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
		int count = 0;
		for (int j = i + 1; j < 9; j++)
			if (n[i] > n[j])
				count++;
		ans += count * jc[9 - i - 1];
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

void prtdfs(state *s)
{
	if (s->father)
		prtdfs(s->father);
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
			cout << s->n[i * 3 + j] << ' ';
		cout << '\n';
	}
	cout << '\n';
}

int cmp(char *c)
{
	for (int i = 0; i < 9; i++)
		if (c[i] != aim[i])
			return 0;
	return 1;
}

int bfs(state sta)
{
	q.push(sta);
	while (!q.empty())
	{
		pastcnt++; //搜索结点数计数器
		state s = q.front();
		q.pop();
		if (cmp(s.n)) //到达目标状态
		{
			cout << "从初始状态到目标状态需要" << s.d << "步\n";
			prtdfs(&s);
			return 1;
		}
		int kt = kangtuo(s.n);
		if (f[kt] == 0) //未到达过此状态
			f[kt] = 1;
		else
			continue; //到达过此状态

		past[cp++] = s;
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
					temp.father = &(past[cp - 1]); //扩展状态的父节点
					q.push(temp); //扩展状态压入open表
				}
			}
	}
	return 0;
}

int main()
{
	state start;
	memset(f, 0, sizeof(f));
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
	if (!bfs(start))
		cout << "该初始状态无解！\n";

	cout <<"已扩展状态："<< pastcnt << '\n';
}
