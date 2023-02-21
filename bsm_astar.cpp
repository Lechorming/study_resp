#include <iostream>
#include <cstring>

using namespace std;

struct state
{
	char n[9]; //用字符数组表示棋盘状态
	int d = -1; //状态深度
	int zx, zy; //空格的位置
	int cost = 0; //代价函数的值
	state *father = NULL; //父节点指针
	state *next = NULL; //open表的next指针
};

int jc[10];			//阶乘
state past[362885]; //记录到达过的状态,以康托展开为顺序的close表
state openhead; //open表表头,在open表中next指针指向下一个cost小于或等于当前状态的状态
char ac[] = "123804765"; //目标状态
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

void fcost(state &s)
{
	int h = 0;
	for (int i = 0; i < 9; i++)
	{
		if (i == 4)
			continue;
		if (s.n[i] != ac[i])
			h++;
	}
	s.cost = s.d + h;
}

int cmp(char *c)
{
	for (int i = 0; i < 9; i++)
		if (c[i] != ac[i])
			return 0;
	return 1;
}

void openpush(state *s)
{
	if(!openhead.next)
	{
		openhead.next=s;
		return;
	}
	if (s->cost < openhead.next->cost)
	{
		s->next = openhead.next;
		openhead.next = s;
		return;
	}
	state *p = openhead.next;
	while (p->next)
	{
		if (s->cost < p->next->cost)
		{
			s->next = p->next;
			p->next = s;
			return;
		}
		p = p->next;
	}
	p->next = s;
}

state opengethead()
{
	state s;
	s = *openhead.next;		//读open表头
	delete openhead.next;	//释放结点
	openhead.next = s.next; //表头指向下一个结点
	return s;
}

void prtans(state *s)
{
	if (s->father)
		prtans(s->father);
	for (int x = 0; x < 3; x++)
	{
		for (int y = 0; y < 3; y++)
			cout << s->n[x * 3 + y] << ' ';
		cout << '\n';
	}
	cout << '\n';
}

int astar(state* start)
{
	openpush(start);
	while (openhead.next)
	{
		pastcnt++; //搜索结点数计数器
		state s;
		s = opengethead(); //获取open表头
		int kt = kangtuo(s.n);
		if (past[kt].d!=-1 && past[kt].d <= s.d) //比较历史状态深度
			continue;
		if (cmp(s.n)) //到达目标状态
		{
			cout << "从初始状态到目标状态需要" << s.d << "步\n";
			prtans(&s);
			return 1;
		}
		past[kt] = s; //更新历史状态

		for (int i = 0; i < 2; i++) //扩展下一层状态
			for (int j = 0; j < 2; j++)
			{
				int x = i ? (j ? s.zx + 1 : s.zx - 1) : s.zx;
				int y = i ? s.zy : (j ? s.zy + 1 : s.zy - 1); // x,y表示扩展状态的空格位置,分别可以向上下左右四个方向扩展
				if (x >= 0 && x <= 2 && y >= 0 && y <= 2 &&
					((s.father ? (x != s.father->zx) : 1) ||
					 (s.father ? (y != s.father->zy) : 1)))
				{
					state *temp = new state; //新建一个状态表示扩展状态
					strcpy(temp->n, s.n); //复制父状态到扩展状态
					swap(temp->n[s.zx * 3 + s.zy], temp->n[x * 3 + y]); //交换空格位置
					temp->d = s.d + 1; //扩展状态深度+1
					temp->zx = x;
					temp->zy = y; //更新扩展状态的空格位置
					temp->father = &(past[kt]); //扩展状态的父节点
					temp->next = NULL; //open表队列的next指针
					fcost(*temp); //计算扩展状态代价函数的值
					openpush(temp); //扩展状态压入open表
				}
			}
	}
	return 0;
}

int main()
{
	state* start=new state;
	cout << "输入初始状态矩阵(用0代替棋盘上的空格):\n";
	for (int i = 0; i < 9; i++)
	{
		cin >> start->n[i];
		if (start->n[i] == '0')
		{
			start->zx = i / 3;
			start->zy = i % 3;
		}
	}
	start->d = 0;
	start->next = NULL;
	jiecheng();
	fcost(*start);

	if(!astar(start))
		cout << "该初始状态无解！\n";

	cout <<"已扩展状态："<< pastcnt << '\n';
}
