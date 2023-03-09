#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 07:12:23 2023

@author: brunobmp
"""



plt.figure(figsize=(8,6))
plt.title("Conv window breaks signal into parts")
lines = [((i*16, 1.2), (i*16, 2.6)) for i in range(512//16)]

for l in lines:
    plt.plot((l[0][0], l[1][0]), (l[0][1], l[1][1]), color="crimson")
    
plt.plot(xi[:512])