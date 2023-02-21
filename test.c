#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <string.h>

int main(void){
    int data[2];
    pipe(data);
    if(fork()==0){
        char command[20];
        sprintf(command,"top -p '%d'",getpid());
        write(data[1],command,strlen(command));
        int num = 1024*1024*10; //每次开辟空间的最小单位10MB
        char *p = (char*)malloc(num);
        for(int i=1;i<20;i++) //循环申请内存，每次都增加申请大小
        {
            p = (char*)realloc(p,num*i);
            sleep(1); //间隔1秒
        }
        free(p);
        printf("[%d] finished!\n",getpid());
        wait(NULL);
        exit(0);
    }
    else{ //子进程执行top命令
        char command[20];
        read(data[0],command,sizeof(command));
        system(command);
        sleep(25);
        exit(0);
    }
}
