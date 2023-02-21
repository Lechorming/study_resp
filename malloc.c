#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct stringdata
{
    char *s;
    int endline;
    struct stringdata *next;
} str;

str *insertst(str *head, char *data)
{
    str *newnode;
    if (!(newnode = malloc(sizeof(str))))
        exit(255);
    if (!(newnode->s = malloc(sizeof(data) + 1)))
        exit(255);
    strcpy(newnode->s, data);
    newnode->endline = (newnode->s[strlen(newnode->s) - 1] == '\n' || newnode->s[strlen(newnode->s) - 1] == '\r');
    newnode->next = NULL;
    str *p = head;
    if (head)
    {
        while (p->next)
            p = p->next;
        p->next = newnode;
    }
    else
        head = newnode;
    return head;
}

void printlist(str *head)
{
    str *p = head;
    int newline = 1;
    int linecount = 0;
    int cnter = 0;
    while (p)
    {
        if (newline)
            printf("第%d行:", ++linecount);
        printf("%s", p->s);
        cnter++;
        newline = p->endline;
        p = p->next;
    }
    printf("共%d行字符串存放在%d个链结点中\n", linecount, cnter);
}

void freelist(str *head)
{
    str *p = head, *n = NULL;
    while (p)
    {
        n = p->next;
        free(p->s);
        free(p);
        p = n;
    }
}

int main()
{
    str *sthead = NULL;
    char s[10];
    printf("请输入字符串,按CTRL+D结束输入\n");
    while (fgets(s, sizeof(s), stdin))
        sthead = insertst(sthead, s);
    printlist(sthead);
    freelist(sthead);
}