ifeq ($(KERNELRELEASE),) 
PWD :=$(shell pwd)
KERSRC := /lib/modules/$(shell uname -r)/build/ 
modules:
	$(MAKE) -C $(KERSRC) M=$(PWD) modules
moules_install:
	$(MAKE) -C $(KERSRC) M=$(PWD) modules_install
.PHONY:
	modules modules_install clean
clean:
	-rm -rf *.o *.cmd.* *.ko
else
modules-objs :=map_driver.o
obj-m := map_driver.o
endif





ifneq ($(KERNELRELEASE),) 
obj-m := proc_helloworld.o
else
KERNELDIR ?= /lib/modules/$(shell uname -r)/build 
PWD := $(shell pwd)
default: 
	$(MAKE) -C $(KERNELDIR) M=$(PWD) modules
clean:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) clean
endif