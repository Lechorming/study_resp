#include<iostream>
#include<cstdio>

#define MAX 1000000

using namespace std;

typedef struct slist_node
{
	int d[3]; //[0]为传教士数；[1]为野人数；[2]=1表示船在左岸，=0表示船在右岸
	int depth;
	slist_node* next = NULL;
}sn;

int n, k;
int mi = 1000000;
sn* head;
sn past[MAX];
int ic = 0;
bool yes = 0;

int count()
{
	sn* p = head;
	int c = 1;
	while (p->next != NULL)
	{
		c++;
		p = p->next;
	}
	return c;
}

void print()
{
	sn* p = head;
	while (1)
	{
        //printf("(%d %d %d)",p->d[0],p->d[1],p->d[2]);
		cout<<'('<<p->d[0]<<' '<<p->d[1]<<' '<<p->d[2]<<')';
		if (p->next != NULL)
		{
			//printf(" -> ");
			cout<<" -> ";
			p = p->next;
		}
		else
		{
			//printf("\n");
			cout<<'\n';
			return;
		}
	}
}

int search_past(sn s)
{
	for (int i = 0; i < ic; i++) //遍历所有曾经状态
	{
		bool dif = 0;
		for (int j = 0; j < 3; j++)
			if (past[i].d[j] != s.d[j])
			{
				dif = 1;
				break;
			}
		if (!dif) //找到相同的曾经状态
			if (past[i].depth >= s.depth)
			{
				past[i].depth = s.depth;
				return -1; //当前状态结点深度更浅或相等
			}
			else
				return 1; //当前状态结点深度更深
	}
	return 0; //没找到相同的曾经状态
}

void record_past(sn s)
{
	for (int i = 0; i < 3; i++)
		past[ic].d[i] = s.d[i];
	past[ic].depth = s.depth;
	ic++;
}

int dfs(sn &s)
{
	if (s.d[0] == 0 && s.d[1] == 0 && s.d[2] == -1) //成功
	{
		int c = count() - 1;
		if (c < mi) mi = c;
		print();
		return 1;
	}

	if ((s.d[0] && s.d[0] < s.d[1]) || (n - s.d[0] && n - s.d[0] < n - s.d[1])) //检测左右岸传教士和野人数是否合法
		return 0;

	int si = search_past(s);
	if (si == 1) return 0; //找到相同的曾经状态且当前状态结点深度更深，返回
	else
		if (si == 0) record_past(s); //没找到相同的曾经状态

	for (int i = 0; i <= (s.d[2] == 1 ? s.d[0] : n - s.d[0]); i++) //传教士
		for (int j = 0; j <= (s.d[2] == 1 ? s.d[1] : n - s.d[1]); j++) //野人
		{
			if (i + j > 0 && i + j <= k && ((i && i >= j) || (i == 0)))
			{
				sn temp;
				temp.d[0] = s.d[0] - s.d[2] * i;
				temp.d[1] = s.d[1] - s.d[2] * j;
				temp.d[2] = -s.d[2];
				temp.depth = s.depth + 1;
				s.next = &temp;
				if (dfs(temp)) yes = 1;
				s.next = NULL;
			}
		}
	return 0;
}

int main()
{
    // ios::sync_with_stdio(0);
    // cin.tie(0);
    // cout.tie(0);
	sn a;
	// printf("输入传教士(野人)数：");
	// scanf("%d", &n);
	// printf("输入船的载人数：");
	// scanf("%d", &k);
	cout << "输入传教士(野人)数：";
	cin >> n;
	cout << "输入船的载人数：";
	cin >> k;
	a.d[0] = a.d[1] = n;
	a.d[2] = 1;
	a.depth = 0;
	head = &a;
	dfs(a);
	if (!yes) cout << "找不到能成功渡河的方案！\n";
	else cout << "最短渡河方案需要" << mi << "步\n";
}