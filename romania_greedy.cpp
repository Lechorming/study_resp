#include<iostream>
#include<cstring>
#include<map>

using namespace std;

int n,m; //城市数和路径数
int a[20][20]; //邻接表
map<string,int> id; //城市名对城市id的映射
map<int,string> name; //城市id对城市名的映射
map<int,int> dis; //城市id对到bucharest的直线距离的映射
int startcity,aimcity; //起始城市和目标城市id
int past[20],pastcnt=0; //记录搜索经过的路径
int totaldis=0; //总路程
bool close[20]; //close表，记录是否到达过该城市

void prtans()
{
    cout<<"从初始城市到目标城市的路径：\n";
    for(int i=0;i<pastcnt-1;i++)
        cout<<name[past[i]]<<"->";
    cout<<name[aimcity]<<'\n';
    cout<<"总路程为："<<totaldis<<"\n经过了"<<pastcnt<<"个城市\n";
}

int greedy()
{
    int now=startcity;
    cout<<"扩展结点和估价函数值：\n";
    while(1)
    {
        past[pastcnt++]=now;
        close[now]=1;
        cout<<name[now]<<' '<<dis[now]<<'\n';
        if(now==aimcity)
        {
            prtans();
            return 1;
        }
        int mindis=1e5;
        int mincity;
        for(int i=0;i<n;i++)
        {
            if(a[now][i]&&dis[i]<mindis&&close[i]==0)
            {
                mindis=dis[i];
                mincity=i;
            }
        }
        totaldis+=a[now][mincity];
        now=mincity;
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
        name[i]=citya;
        dis[i]=temp;
    }
    for(int i=0;i<m;i++)
    {
        cin>>citya>>cityb>>temp;
        a[id[citya]][id[cityb]]=temp;
        a[id[cityb]][id[citya]]=temp;
    }
    startcity=id["zerind"];
    aimcity=id["bucharest"];

    if(!greedy())
        cout<<"无解";
}