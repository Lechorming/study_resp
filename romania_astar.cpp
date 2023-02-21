#include<iostream>
#include<cstring>
#include<map>
#include<queue>

using namespace std;

struct node
{
    string name; //城市名
    int cost=-1; //代价函数值
    int nowdis=0; //从初始城市到当前城市的总路程
    int father=-1; //父节点
    friend bool operator < (node a,node b) //用于优先队列比较,
    {
        return a.cost>b.cost;
    }
};

int n,m; //城市数和路径数
int a[20][20]; //邻接表
map<string,int> id; //城市名对城市id的映射
map<int,int> dis; //城市id对到bucharest的直线距离的映射
node startcity; //起始城市
int aimcity; //目标城市id
int totaldis=0; //总路程
node close[20]; //close表，记录是否到达过该城市
priority_queue<node> open; //open表,优先队列按cost从小到大排列
int pastcnt=0;

void prtans(int id)
{
    if(close[id].father!=-1)
        prtans(close[id].father);

    if(id!=aimcity)
        cout<<close[id].name<<"->";
    else
        cout<<close[id].name<<'\n';

}

int fcost(int id,int nowdis)
{
    return nowdis+dis[id];
}

int astar()
{
    open.push(startcity);
    cout<<"扩展结点和估价函数值：\n";
    while(!open.empty())
    {
        pastcnt++;
        node now=open.top();
        open.pop();
        cout<<now.name<<' '<<now.cost<<'\n';
        if(close[id[now.name]].cost!=-1&&
            close[id[now.name]].cost<=now.cost)
            continue;

        close[id[now.name]]=now;

        if(id[now.name]==aimcity)
        {
            cout<<"从初始城市到目标城市的路径：\n";
            prtans(id[now.name]);
            cout<<"总路程为："<<now.nowdis<<"\n扩展了"<<pastcnt<<"个城市\n";
            return 1;
        }

        for(int i=0;i<n;i++)
        {
            if(a[id[now.name]][i])
            {
                node temp;
                temp.name=close[i].name;
                temp.nowdis=now.nowdis+a[id[now.name]][i];
                temp.cost=fcost(i,temp.nowdis);
                temp.father=id[now.name];
                open.push(temp);
            }
        }
        
    }
    return 0;
}

int main()
{
    memset(a,0,sizeof(a));
    cin>>n>>m;
    string citya,cityb;
    int temp;
    for(int i=0;i<n;i++)
    {
        cin>>citya>>temp;
        id[citya]=i;
        close[i].name=citya;
        dis[i]=temp;
    }
    for(int i=0;i<m;i++)
    {
        cin>>citya>>cityb>>temp;
        a[id[citya]][id[cityb]]=temp;
        a[id[cityb]][id[citya]]=temp;
    }
    startcity.name="zerind";
    startcity.cost=fcost(id[startcity.name],startcity.nowdis);
    startcity.father=-1;

    aimcity=id["bucharest"];

    if(!astar())
        cout<<"无解";
}