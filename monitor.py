#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Rishab Pal'

import psutil
# To install the package for the below import
# pip install nvidia-ml-py3
import nvidia_smi


class CheckHardwareUtilization(object):
    def __init__(self):
        self.cpu_usage = psutil.cpu_percent()
        self.ram_usage = psutil.virtual_memory()
        nvidia_smi.nvmlInit()
        self.gpu_device_count = nvidia_smi.nvmlDeviceGetCount()
    
    def cpu(self):
        # print(f"CPU usage: {self.cpu_usage}%")
        return self.cpu_usage
       
    def ram(self):
        ram_used = self.ram_usage.used/(1024.0**3)
        total_ram = self.ram_usage.total/(1024.0**3)
        ram_used_per = (ram_used/total_ram)*100
        # print(f"Ram usage: {ram_used_per}%")
        return ram_used_per
        
    def gpu(self):
        gpu_usage = []
        for i in range(self.gpu_device_count):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            # print(f'gpu: {res.gpu}%, gpu-mem: {res.memory}%')
            gpu_usage.append({'id': i, 'gpu': res.gpu, 'gpu_mem': res.memory})
        return gpu_usage
