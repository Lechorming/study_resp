#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
int main()
{
    int p;
    p = fork();
    switch (p)
    {
    case -1:
        printf("创建子进程失败！");
        exit(1);
    case 0:
        printf("子进程pid为：%d\n", getpid());
        while (1);
        exit(1);
    default:
        printf("父进程pid为：%d\n", getpid());
        while (1);
        exit(0);
    }
}