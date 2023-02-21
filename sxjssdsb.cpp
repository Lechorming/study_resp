#include <iostream>
#include <cstring>
#include <stack>
#include <map>

using namespace std;
// LL(1)分析表
string table[5][8] =
    {
        {"TD", "error", "error", "error", "error", "TD", "error", "error"},
        {"error", "+TD", "-TD", "error", "error", "error", "@", "@"},
        {"FU", "error", "error", "error", "error", "FU", "error", "error"},
        {"error", "@", "@", "*FU", "/FU", "error", "@", "@"},
        {"i", "error", "error", "error", "error", "(E)", "error", "error"}};
char S = 'E'; // 文法开始符号
map<char, int> terminal; // 终结符映射表
map<char, int> non_terminal; // 非终结符映射表
string words[100]; // 单词字符串数组
int wordsv[100]; // 单词符号种别编码数组
int wordslen; // 单词数

void printword(string word, int value)
{
    cout << "(" << word << "," << value << ")\n";
}

bool wordanalyze(string inputstring)
{
    string s1;
    int strp = 0, inplen = inputstring.length(), wordsp = 0;
    while (strp < inplen)
    {
        string temp;
        char c = inputstring[strp++];
        if (c >= '0' && c <= '9')
        { // 数字
            while (1)
            {
                temp += c;
                if (inputstring[strp] >= '0' && inputstring[strp] <= '9')
                    c = inputstring[strp++];
                else
                    break;
            }
            words[wordsp] = temp;
            wordsv[wordsp++] = terminal['i'];
            printword(temp, terminal['i']);
        }
        else
        {                                           // 符号
            if (terminal.find(c) == terminal.end()) // 不存在该符号
                return 0;
            temp += c;
            words[wordsp] = temp;
            wordsv[wordsp++] = terminal[c];
            printword(temp, terminal[c]);
        }
    }
    wordslen = wordsp;
    return 1;
}

bool isterminal(char c)
{
    if (terminal.end() == terminal.find(c))
        return 0;
    else
        return 1;
}

void printstack(stack<char> s)
{
    while (!s.empty())
    {
        cout << s.top();
        s.pop();
    }
    cout << "\t\t";
}

void printstr(int p)
{
    for (int i = p; i < wordslen; i++)
        cout << words[i];
    cout << "\t\t";
}

bool grammaanalyze()
{
    stack<char> sign;
    sign.push('#');
    sign.push(S);
    int wordsp = 0;
    int v = wordsv[wordsp++];
    bool flag = true;
    int count = 0;
    cout << "步数\t\t符号栈\t\t字符\t\t使用文法\n";
    while (flag)
    {
        cout << ++count << "\t\t";
        printstack(sign);
        printstr(wordsp - 1);
        char x = sign.top();
        sign.pop();
        if (isterminal(x))
        { // x是终结符
            if (terminal[x] == v)
            {
                cout << x << "匹配";
                if (x == '#') // 结束
                    flag = false;
                else
                    v = wordsv[wordsp++];
            }
            else
            {
                cout << "错误!终结符" << x << "和" << words[wordsp - 1] << "不匹配";
                return false;
            }
        }
        else
        { // 如果是非终结符，需要移进操作
            string right = table[non_terminal[x]][v];
            if (right != "error")
            {
                cout << x << "->" << right;
                if (right != "@") // 文法右边不为空串
                    for (int i = right.length() - 1; i >= 0; i--)
                        sign.push(right[i]);
            }
            else
            { //不匹配
                cout << "没有使" << x << "与" << words[wordsp - 1] << "配对的文法!\n";
                return false;
            }
        }
        cout << '\n';
    }
    return true;
}

void init()
{
    non_terminal['E'] = 0;
    non_terminal['D'] = 1;
    non_terminal['T'] = 2;
    non_terminal['U'] = 3;
    non_terminal['F'] = 4;
    terminal['i'] = 0;
    terminal['+'] = 1;
    terminal['-'] = 2;
    terminal['*'] = 3;
    terminal['/'] = 4;
    terminal['('] = 5;
    terminal[')'] = 6;
    terminal['#'] = 7;
}

int main()
{
    string inputstr;
    cout << "输入字符串(以#结束):";
    cin >> inputstr;
    init();
    if (wordanalyze(inputstr))
    {
        cout << "词法分析成功\n";
        if (grammaanalyze())
            cout << "数学算式识别成功\n";
        else
            cout << "语法识别失败\n";
    }
    else
        cout << "词法分析失败\n";
}