#include<iostream>
#include<cstring>
#include<stack>
#include<map>

using namespace std;

string table[5][6]=
{
    {"TD","error","error","TD","error","error"},
    {"error","+TD","error","error","@","@"},
    {"FU","error","error","FU","error","error"},
    {"error","@","*FU","error","@","@"},
    {"i","error","error","(E)","error","error"}
};
char S='E'; //文法开始符号
map<char,int> terminal;
map<char,int> non_terminal;

bool isterminal(char c)
{
    if(terminal.end()==terminal.find(c))
        return 0;
    else
        return 1;
}

void printstack(stack<char> s)
{
    while(!s.empty())
    {
        cout<<s.top();
        s.pop();
    }
    cout<<"\t\t";
}

void printstr(string s,int p)
{
    for(int i=p;i<s.length();i++) cout<<s[i];
    cout<<"\t\t";
}

bool analyze(string inputstring)
{
    stack<char> sign;
    sign.push('#');
    sign.push(S);
    int strp = 0;
    char a = inputstring[strp++];
    bool flag = true;
    int count=0;
    cout<<"步数\t\t符号栈\t\t字符\t\t使用文法\n";
    while (flag)
    {
        cout<<++count<<"\t\t";
        printstack(sign);
        printstr(inputstring,strp-1);
        char x = sign.top();
        sign.pop();
        if (isterminal(x))
        { //x是终结符
            if (x == a)
            {
                cout<<x<<"匹配";
                if (x == '#') //结束
                    flag = false;
                else
                    a = inputstring[strp++];
            }
            else
            {
                cout<<"错误!终结符"<<x<<"和"<<a<<"不匹配";
                return false;
            }
        }
        else
        { //如果是非终结符，需要移进操作
            string right = table[non_terminal[x]][terminal[a]];
            if (right != "error")
            {
                cout<<x<<"->"<<right<<'\t';
                if(right!="@")
                    for (int i = right.length() - 1; i >= 0; i--)
                        sign.push(right[i]);
            }
            else
            {
                cout<<"没有使"<<x<<"与"<<a<<"配对的文法!\n";
                return false;
            }
        }
        cout<<'\n';
    }
    return true;
}

void init()
{
    non_terminal['E']=0;
    non_terminal['D']=1;
    non_terminal['T']=2;
    non_terminal['U']=3;
    non_terminal['F']=4;
    terminal['i']=0;
    terminal['+']=1;
    terminal['*']=2;
    terminal['(']=3;
    terminal[')']=4;
    terminal['#']=5;

}

int main()
{
    string inputstr;
    cout<<"输入字符串(以#结束):";
    cin>>inputstr;
    init();
    if(analyze(inputstr))
        cout<<"文法识别成功\n";
    else
        cout<<"文法识别失败\n";
}