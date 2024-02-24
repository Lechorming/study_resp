#include<iostream>


using namespace std;

class Students{
public:
    string sname;
    int sno;
    int sde;
   
};

void swap(class Students &a,class Students &b){
    class Students temp;
    temp.sname=a.sname;
    temp.sno=a.sno;
    temp.sde=a.sde;
    a.sname=b.sname;
    a.sno=b.sno;
    a.sde=b.sde;
    b.sname=temp.sname;
    b.sno=temp.sno;
    b.sde=temp.sde;
}

void Ascend(class Students* t,int num){
    for(int i=0;i<num;i++){
        for(int j=0;j<num;j++){
            if(t[i].sde>=t[j].sde){
                if(t[i].sde==t[j].sde && t[i].sno>t[j].sno)
                    swap(t[i],t[j]);
            }
        }
    }
    return;

}

void Decend(class Students* t,int num){
    for(int i=0;i<num;i++){
        for(int j=0;j<num;j++){
            if(t[i].sde<=t[j].sde){
                if(t[i].sde==t[j].sde && t[i].sno<t[j].sno)
                    swap(t[i],t[j]);
            }
        }
    }
    return;
}

void Show(class Students* t,int num){
    for(int i=0;i<num;i++)
        cout<<t[i].sname<<','<<t[i].sno<<','<<t[i].sde<<'\n';
    return;
}




int main(){
    Students table[100];

    string temp;
    string name;
    int no,de;
    cin>>temp;
    int i=0;
    while(temp!="*"){
        int cnt=0;
        int idx;
        for(int i=0;i<temp.length();i++){
            if(temp[i]==',') {
                if(cnt==0){
                    name=temp.substr(0,i);
                    cnt++;
                    idx=i;
                }
                if(cnt==1){
                    no=stoi(temp.substr(idx+1,i-idx));
                    cnt++;
                    idx=i;
                }
            }
        }
        de=stoi(temp.substr(idx+1,temp.length()-idx));
        table[i].sname=name;
        table[i].sno=no;
        table[i].sde=de;
        i++;
    }
    Ascend(table,i);
    Show(table,i);
    Decend(table,i);
    Show(table,i);
    

}