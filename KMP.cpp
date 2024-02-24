#include<iostream>
#include<string>
using namespace std;

int* buildnext(string s){
    int* next=new int[s.length()];
    next[0]=-1;
    int j=0,k=-1;
    while(j<s.length()-1){
        if(k==-1||s[j]==s[k]){
            k++;
            j++;
            next[j]=k;
        }
        else{
            k=next[k];
        }
    }
    return next;
}

bool KMP(string s1,string s2){
    int* next=buildnext(s2);
    int i=0,j=0;

    while(i<s1.length() && j<s2.length())
    {
        if(j==(-1)||s1[i]==s2[j]){
            i++;
            j++;
        }
        else{
            j=next[j];
        }
    }
    
    if(j==s2.length()) 
        return 1;
    else 
        return 0;
}