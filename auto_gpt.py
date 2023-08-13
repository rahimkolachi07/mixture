#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
import streamlit as st
from langchain.llms import OpenAI
st.title("my youtube gpt")
prompt=st.text_input("your prompt")


# In[5]:


streamlit run auto gpt


# In[ ]:




