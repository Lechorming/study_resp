#include <linux/init.h>
#include <linux/kernel.h>
#include <linux/proc_fs.h>
#include <linux/module.h>
#include <linux/sched.h>
#include <asm/uaccess.h>

static struct proc_dir_entry *proc_parent;
int len, temp;
char *msg; //字符数组

static ssize_t read_proc(struct file *filp, char __user *buf, size_t count, loff_t *offp)
{
    if (count > temp)
        count = temp;
    temp = temp - count;

    copy_to_user(buf, msg, count);
    if (count == 0)
        temp = len;
    return count;
}

static const struct file_operations proc_fops = {
    .read = read_proc
};

void create_new_proc_entry(void)
{
    proc_parent = proc_mkdir("hello", NULL); //创建一个目录hello

    if (!proc_parent) //创建目录失败
        printk(KERN_INFO "Error creating proc entry");

    proc_create("world", 0, proc_parent, &proc_fops);

    msg = "hello world\n"; //文件内容
    len = strlen(msg);     //获得字符串msg的长度
    temp = len;
    printk(KERN_INFO "len = %d", len);
    printk(KERN_INFO "proc initialized");
}

int proc_init(void) //模块的初始化函数
{
    create_new_proc_entry();
    return 0;
}

void proc_cleanup(void) //模块的退出和清理函数
{
    printk(KERN_INFO "Inside cleanup_module\n");
    remove_proc_entry("hello", proc_parent); //移除目录hello
    remove_proc_entry("world", NULL);        //移除文件world
}

module_init(proc_init);
module_exit(proc_cleanup);
MODULE_LICENSE("GPL");
