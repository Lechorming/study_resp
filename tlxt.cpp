#include <iostream>
#include <cstdio>
#include <string>
#include <fstream>

using namespace std;

string database[1000]; //事实数据库
int dblen = 0;        //数据库大小
struct premises       //前提链表
{
    int data; //对应事实数据库下标
    premises *next = NULL;
};
struct rules //规则结构体
{
    premises *premise = NULL; //前提
    int plen = 0;             //前提个数
    int conclusion;           //结论
} r[100];
int rlen = 0; //规则数

int searchdb(string s) //查找数据库元素，存在则返回下标，不存在则返回-1
{
    for (int i = 0; i < dblen; i++)
        if (s == database[i])
            return i;
    return -1;
}

void insertpre(premises *&tail, int &data, premises *&head) //规则前提链表插入
{
    premises *p = new premises;
    p->data = data;
    if (tail == NULL)
        head = p;
    else
        tail->next = p;
    tail = p;
}

void initdatabase() //初始化数据库
{
    ifstream fin;
    fin.open("rules.txt");
    string temp;
    int res;
    while (fin >> temp) //读取结论,构建规则和数据库
    {
        res = searchdb(temp); //在数据库中寻找是否已存在
        if (res < 0)          //不存在，添加进数据库
        {
            database[dblen] = temp;
            r[rlen].conclusion = dblen;
            dblen++;
        }
        else
        {
            r[rlen].conclusion = res;
        }
        string pre;
        premises *old = NULL;
        while (fin >> pre) //读取前提
        {
            if (pre != "*")
            {
                res = searchdb(pre); //在数据库中寻找
                if (res < 0)         //数据库中没有,添加进数据库
                {
                    database[dblen] = pre;
                    insertpre(old, dblen, r[rlen].premise);
                    dblen++;
                }
                else //数据库中已有
                {
                    insertpre(old, res, r[rlen].premise);
                }
                r[rlen].plen++;
            }
            else //读完一条规则
            {
                rlen++;
                old = NULL;
                break;
            }
        }
    }
    fin.close();
}

bool matching(premises *rulehead, premises *inhead) //判断规则中的前提和特征链表是否匹配
{
    bool find = 0, match = 1;
    premises *rulep = rulehead, *inp = inhead;
    while (rulep)
    {
        find = 0;
        inp = inhead;
        while (inp)
        {
            if (rulep->data == inp->data)
            {
                find = 1;
                break;
            }
            inp = inp->next;
        }
        if (!find)
        {
            match = 0;
            break;
        }
        rulep = rulep->next;
    }
    return match;
}

void printrule(rules ru, int c) //打印规则
{
    premises *p = ru.premise;
    cout << c << '.';
    while (p)
    {
        cout << database[p->data];
        p = p->next;
        if (p)
            cout << ',';
        else
            cout << "-->";
    }
    cout << database[ru.conclusion] << '\n';
}

//删除特征链表中和规则中前提相同的项
void deletepre(premises *rulehead, premises *&inhead, int &inplen, premises *&tail)
{
    premises *rulep = rulehead, *inp = inhead, *last = NULL;
    while (rulep)
    {
        while (inp)
        {
            if (rulep->data == inp->data)
            {
                if (last)
                    last->next = inp->next;
                else
                    inhead = inp->next;
                if (tail == inp)
                    tail = last;
                delete inp;
                inplen--;
                break;
            }
            last = inp;
            inp = inp->next;
        }
        rulep = rulep->next;
        inp = inhead;
        last = NULL;
    }
}

void printresult(bool result, premises *p) //打印正向推理的结果
{
    if (result)
        cout << "推理成功,该动物是:" << database[p->data] << '\n';
    else
    {
        cout << "推理失败,找不到匹配该特征的动物:";
        while (p)
        {
            cout << database[p->data];
            p = p->next;
            if (p)
                cout << ',';
            else
                cout << '\n';
        }
    }
}

void positive(premises *&head, int plen, premises *&tail) //正向推理
{
    int rucount = 0;
    bool match = 0;
    cout << "---------------使用规则----------------\n";
    while (1)
    {
        match = 0;
        for (int i = 0; i < rlen; i++) //遍历规则与特征链表进行匹配
        {
            if (r[i].plen <= plen)                        //规则中前提项数大于特征链表长度的直接跳过
                if (match = matching(r[i].premise, head)) //匹配判断
                {
                    rucount++;
                    deletepre(r[i].premise, head, plen, tail); //删除链表中匹配到的项
                    insertpre(tail, r[i].conclusion, head);    //插入规则中的结论到链表中
                    plen++;
                    printrule(r[i], rucount); //打印规则
                    break;
                }
        }
        if (!match || plen == 1) //没有匹配的规则或特征链表长度等于1就退出循环
            break;
    }
    cout << "---------------------------------------\n";
    printresult(match, head); //打印结果
}

void ne_printresult(premises *p) //反向推理打印结果
{
    cout << "匹配结果:";
    while (p)
    {
        cout << database[p->data];
        p = p->next;
        if (p)
            cout << ',';
        else
            cout << '\n';
    }
}

void negative(int animal) //反向推理
{
    premises *head = NULL, *tail = NULL; //创建一个结果链表
    insertpre(tail, animal, head);       //将动物的数据插入链表
    bool match = 0;                      //匹配标志
    premises *p = head;                  //链表指针
    int plen = 1, rucount = 0;           //链表长度,使用了的规则数计数器
    cout << "---------------使用规则----------------\n";
    for (int i = 0; i < rlen; i++) //遍历规则
    {
        match = 0;
        p = head;
        while (p) //遍历特征链表
        {
            if (r[i].conclusion == p->data) //判断规则中结论是否与链表中某一项相同
            {
                premises temp;
                temp.data = r[i].conclusion;
                deletepre(&temp, head, plen, tail); //删除链表中与规则结论匹配的一项
                premises *rup = r[i].premise;
                while (rup) //遍历匹配的规则的前提并插入链表
                {
                    insertpre(tail, rup->data, head);
                    plen++;
                    rup = rup->next;
                }
                match = 1;
                rucount++;
                printrule(r[i], rucount); //打印使用了的规则
                break;
            }
            p = p->next;
        }
        if (match) //存在匹配的规则就要重新遍历规则表
            i = -1;
    }
    cout << "-----------------------------------\n";
    ne_printresult(head); //打印反向推理的结果
}

int main()
{
    initdatabase(); //初始化数据库
    cout << "---------------数据库----------------\n";
    for (int i = 0; i < dblen; i++)
    {
        cout << i << "." << database[i] << "\t";
        if ((i + 1) % 6 == 0)
            cout << '\n';
    }
    cout << "\n-------------------------------------\n";
    while (1)
    {
        cout << "请选择推理方法编号(1.正向推理;2.反向推理;3.退出):";
        int t;
        cin >> t;
        if (t == 1)
        {
            cout << "请输入特征编号(以*结束):";
            premises *p = NULL; //用链表存储特征
            premises *tail = NULL;
            int plen = 0;
            string temp;
            while (1) //循环读取数据,直到读取到*号
            {
                cin >> temp;
                if (temp == "*")
                    break;
                int data = stoi(temp.c_str());
                insertpre(tail, data, p);
                plen++;
            }
            positive(p, plen, tail); //正向推理
        }
        else if (t == 2)
        {
            cout << "请输入动物编号:";
            int animal;
            cin >> animal;
            negative(animal); //反向推理
        }
        else
            break;
    }
}
