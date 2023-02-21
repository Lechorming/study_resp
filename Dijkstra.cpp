#include<iostream>
#include<fstream>
#include<string>
#include<cstring>
#include<queue>

using namespace std;

void printopenclose(int len,queue<string> open,queue<string> close)
{
    cout<<len<<';';
    while(!open.empty())
    {
        string s=open.front();
        cout<<'('<<s[0]<<','<<s[1]<<')';
        open.pop();
        if(!open.empty()) cout<<',';
    }
    cout<<';';
    while(!close.empty())
    {
        string s=close.front();
        cout<<'('<<s[0]<<','<<s[1]<<')';
        close.pop();
        if(!close.empty()) cout<<',';
    }
    cout<<'\n';
}

void printpath(int x,int y,string path[10][10])
{
    if(x==0&&y==0)
    {
        cout<<'('<<x<<','<<y<<")->";
        return;
    }
    string s=path[x][y];
    int nx=s[0]-'0',ny=s[1]-'0';
    printpath(nx,ny,path);
    cout<<'('<<x<<','<<y<<')';
    if(x!=9||y!=9) cout<<"->";
    else cout<<'\n';
}

void Dijkstra(int dist[10][10])
{
    queue<string> open;
    queue<string> close;
    int lastlen=-1,nx,ny;
    bool visited[10][10];
    string path[10][10];
    visited[0][0]=1;
    path[0][0]="00";
    open.push("00");
    while(!open.empty())
    {
        string s=open.front();
        int x=s[0]-'0',y=s[1]-'0';
        //输出open表close表
        if(dist[x][y]>lastlen)
        {
            lastlen=dist[x][y];
            printopenclose(lastlen,open,close);
        }
        //到达终点输出路径
        if(x==9&&y==9)
        {
            cout<<"最短路径长度:"<<dist[x][y]<<'\n';
            printpath(9,9,path);
            return;
        }
        //更新open表
        for(int i=0;i<4;i++)
        {
            switch(i)
            {
                case 0:
                    nx=x-1;
                    ny=y;
                    break;
                case 1:
                    nx=x+1;
                    ny=y;
                    break;
                case 2:
                    nx=x;
                    ny=y-1;
                    break;
                case 3:
                    nx=x;
                    ny=y+1;
                    break;
            }
            if(nx<0||nx>9||ny<0||ny>9) continue;
            if(dist[nx][ny]!=-1&&!visited[nx][ny])
            {
                dist[nx][ny]=dist[x][y]+1;
                visited[nx][ny]=1;
                char cnx=nx+'0',cny=ny+'0';
                string ns="";
                ns+=cnx; ns+=cny;
                open.push(ns);
                path[nx][ny]=s;
            }
        }
        open.pop();
        close.push(s);
    }

}

struct opennode
{
    int pos;
    int f;
    opennode* next=NULL;
}*openhead=NULL;

void popopen()
{
    if(openhead)
    {
        opennode* p=openhead;
        openhead=openhead->next;
        delete p;
    }
}

void insertopen(opennode *newnode)
{
    if(openhead)
    {
        opennode *p=openhead,*last=NULL;
        while(p)
        {
            //f递增排列
            if((p->f)>=(newnode->f))
            {
                //若p为openhead要特殊处理
                if(p==openhead)
                {
                    newnode->next=p;
                    openhead=newnode;
                    return;
                }
                else
                {
                    last->next=newnode;
                    newnode->next=p;
                    return;
                }
            }
            last=p;
            p=p->next;
        }
        last->next=newnode;
    }
    else
    {
        openhead=newnode;
    }
}

int f(int cost,int x,int y)
{
    return cost+18-x-y;
}

void printastar(int c,opennode* p,queue<int> close)
{
    cout<<c<<';';
    while(p)
    {
        cout<<'('<<(p->pos)/10<<','<<(p->pos)%10<<")";
        if(p->next) cout<<',';
        p=p->next;
    }
    cout<<';';
    while(!close.empty())
    {
        int s=close.front();
        cout<<'('<<s/10<<','<<s%10<<')';
        close.pop();
        if(!close.empty()) cout<<',';
    }
    cout<<'\n';
}

void printastarpath(int x,int y,int path[10][10])
{
    if(x==0&&y==0)
    {
        cout<<'('<<x<<','<<y<<")->";
        return;
    }
    int s=path[x][y];
    int nx=s/10,ny=s%10;
    printastarpath(nx,ny,path);
    cout<<'('<<x<<','<<y<<')';
    if(x!=9||y!=9) cout<<"->";
    else cout<<'\n';
}

void Astar(int a[10][10])
{
    //初始化
    openhead=new opennode;
    openhead->pos=00;
    openhead->f=f(0,0,0);
    queue<int> close;
    int path[10][10],count=0,nx,ny;
    bool visited[10][10];
    visited[0][0]=1;
    path[0][0]=0;
    a[0][0]=0;
    while(openhead)
    {
        int x=(openhead->pos)/10,y=(openhead->pos)%10;
        //输出两个表
        printastar(count++,openhead,close);
        // for(int i=0;i<10;i++)
        // {
        //     for(int j=0;j<10;j++)
        //         cout<<a[i][j]<<'\t';
        //     cout<<'\n';
        // }
        popopen();
        //到达终点
        if(x==9&&y==9)
        {
            cout<<"最短路径长度:"<<a[x][y]<<'\n';
            printastarpath(9,9,path);
            return;
        }
        //更新open表
        for(int i=0;i<4;i++)
        {
            switch(i)
            {
                case 0:
                    nx=x-1;
                    ny=y;
                    break;
                case 1:
                    nx=x+1;
                    ny=y;
                    break;
                case 2:
                    nx=x;
                    ny=y-1;
                    break;
                case 3:
                    nx=x;
                    ny=y+1;
                    break;
            }
            if(nx<0||nx>9||ny<0||ny>9) continue;
            if(a[nx][ny]!=-1&&!visited[nx][ny])
            {
                if((a[nx][ny]>a[x][y]+1)||(a[nx][ny]==0)) a[nx][ny]=a[x][y]+1;
                visited[nx][ny]=1;
                opennode *newnode=new opennode;
                newnode->pos=nx*10+ny;
                newnode->f=f(a[x][y]+1,nx,ny);
                insertopen(newnode);
                path[nx][ny]=x*10+y;
            }
        }
        close.push(x*10+y);
    }
    cout<<"无解\n";
}

int main()
{
    ifstream fin;
    fin.open("Dijkstra.in");
    int a[10][10];
    memset(a,0,sizeof(a));
    int n,x,y;
    fin>>n;
    while(n--)
    {
        fin>>x>>y;
        a[x][y]=-1;
    }


    //Dijkstra(a);
    Astar(a);


    fin.close();
}