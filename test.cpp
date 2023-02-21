#include<cstring>
#include<iostream>
using namespace std;
int main()
{
    char c[3];
    const char a[]={'a','b','c'};
    cout<<"输入\n";
    for(int i;i<3;i++)
    cin>>c[i];
    cout<<strcmp(c,a)<<'\n';
    return 0;
}