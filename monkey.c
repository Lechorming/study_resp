#include<stdio.h>

struct state
{
	int monkeypos;
	int monkeyonbox;
	int boxpos;
	int bananapos;
	int getbanana;
};

struct state status;

void walk(int n)
{
	status.monkeypos=n;
	printf("猴子移动到了%c处\n",(char)(n+'a'));
}

void movebox(int n)
{
	status.monkeypos=n;
	status.boxpos=n;
	printf("猴子移动箱子到了%c处\n",(char)(n+'a'));
}

void climb()
{
	status.monkeyonbox=1;
	printf("猴子爬上箱子\n");
}

void get()
{
	status.getbanana=1;
	printf("猴子拿到香蕉了\n");
}


int main()
{
	while(1)
	{
		printf("0代表a,1代表b,2代表c\n请输入猴子初始位置(退出输入-1):");
		scanf("%d",&status.monkeypos);
		if(status.monkeypos==-1) break;
		else
		{
			printf("请输入箱子初始位置:");
			scanf("%d",&status.boxpos);
			printf("请输入香蕉位置:");
			scanf("%d",&status.bananapos);
			status.getbanana=0;
			status.monkeyonbox=0;
		}
		while(1)
		{
			if(status.monkeypos!=status.boxpos)
				walk(status.boxpos);
			else
				if(status.boxpos!=status.bananapos)
					movebox(status.bananapos);
				else
					if(!status.monkeyonbox)
						climb();
					else
						if(!status.getbanana)
							get();
						else
							break;	
		}
		printf("已完成！\n进入下一轮...\n");
	}
	printf("已退出！\n");
		
	return 0;	
		
}